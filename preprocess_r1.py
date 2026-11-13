#!/usr/bin/env python3
"""Extract session-level R1 features from PCAPNG files and write clean CSV splits."""

from __future__ import annotations

import argparse
import csv
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


FEATURE_COLUMNS = (
    "bytes_sent",
    "bytes_received",
    "packets_sent",
    "packets_received",
    "mean_iat_seconds",
    "std_iat_seconds",
    "duration_seconds",
    "label",
)


@dataclass(frozen=True)
class Interface:
    link_type: int
    timestamp_resolution: float
    timestamp_offset: int


@dataclass(frozen=True)
class Packet:
    timestamp: float
    source: tuple[bytes, int]
    destination: tuple[bytes, int]
    protocol: int
    length: int


class Session:
    def __init__(self, packet: Packet) -> None:
        self.initiator = packet.source
        self.first_timestamp = packet.timestamp
        self.last_timestamp = packet.timestamp
        self.min_timestamp = packet.timestamp
        self.max_timestamp = packet.timestamp
        self.bytes_sent = 0
        self.bytes_received = 0
        self.packets_sent = 0
        self.packets_received = 0
        self.iat_count = 0
        self.iat_mean = 0.0
        self.iat_m2 = 0.0
        self.add(packet)

    def add(self, packet: Packet) -> None:
        if packet.timestamp > self.last_timestamp:
            gap = packet.timestamp - self.last_timestamp
            self.iat_count += 1
            delta = gap - self.iat_mean
            self.iat_mean += delta / self.iat_count
            self.iat_m2 += delta * (gap - self.iat_mean)
            self.last_timestamp = packet.timestamp

        self.min_timestamp = min(self.min_timestamp, packet.timestamp)
        self.max_timestamp = max(self.max_timestamp, packet.timestamp)
        if packet.source == self.initiator:
            self.bytes_sent += packet.length
            self.packets_sent += 1
        else:
            self.bytes_received += packet.length
            self.packets_received += 1

    def as_row(self, label: str) -> dict[str, int | float | str]:
        std_iat = (
            math.sqrt(self.iat_m2 / self.iat_count) if self.iat_count else 0.0
        )
        return {
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "packets_sent": self.packets_sent,
            "packets_received": self.packets_received,
            "mean_iat_seconds": self.iat_mean if self.iat_count else 0.0,
            "std_iat_seconds": std_iat,
            "duration_seconds": self.max_timestamp - self.min_timestamp,
            "label": label,
        }


def _read_options(data: bytes, endian: str, offset: int) -> Iterator[tuple[int, bytes]]:
    while offset + 4 <= len(data):
        code, length = struct.unpack_from(endian + "HH", data, offset)
        offset += 4
        if code == 0:
            return
        end = offset + length
        if end > len(data):
            raise ValueError("PCAPNG option extends beyond its containing block")
        yield code, data[offset:end]
        offset += (length + 3) & ~3


def _iter_pcapng_blocks(path: Path) -> Iterator[tuple[int, str, bytes]]:
    endian: str | None = None
    with path.open("rb") as capture:
        while True:
            block_type = capture.read(4)
            if not block_type:
                return
            if len(block_type) != 4:
                raise ValueError(f"{path}: truncated PCAPNG block type")

            if block_type == b"\x0a\x0d\x0d\x0a":
                length_bytes = capture.read(4)
                byte_order_magic = capture.read(4)
                if len(length_bytes) != 4 or len(byte_order_magic) != 4:
                    raise ValueError(f"{path}: truncated PCAPNG section header")
                if byte_order_magic == b"\x4d\x3c\x2b\x1a":
                    endian = "<"
                elif byte_order_magic == b"\x1a\x2b\x3c\x4d":
                    endian = ">"
                else:
                    raise ValueError(f"{path}: invalid PCAPNG byte-order magic")

                total_length = struct.unpack(endian + "I", length_bytes)[0]
                if total_length < 28 or total_length % 4:
                    raise ValueError(f"{path}: invalid PCAPNG section length")
                remainder = capture.read(total_length - 12)
                if len(remainder) != total_length - 12:
                    raise ValueError(f"{path}: truncated PCAPNG section header")
                if struct.unpack(endian + "I", remainder[-4:])[0] != total_length:
                    raise ValueError(f"{path}: mismatched PCAPNG block lengths")
                yield 0x0A0D0D0A, endian, b""
                continue

            if endian is None:
                raise ValueError(f"{path}: missing PCAPNG section header")
            length_bytes = capture.read(4)
            if len(length_bytes) != 4:
                raise ValueError(f"{path}: truncated PCAPNG block length")
            total_length = struct.unpack(endian + "I", length_bytes)[0]
            if total_length < 12 or total_length % 4:
                raise ValueError(f"{path}: invalid PCAPNG block length")
            remainder = capture.read(total_length - 8)
            if len(remainder) != total_length - 8:
                raise ValueError(f"{path}: truncated PCAPNG block")
            if struct.unpack(endian + "I", remainder[-4:])[0] != total_length:
                raise ValueError(f"{path}: mismatched PCAPNG block lengths")
            yield struct.unpack(endian + "I", block_type)[0], endian, (
                block_type + length_bytes + remainder
            )


def _packet_from_frame(frame: bytes, link_type: int, timestamp: float) -> Packet | None:
    if link_type == 1:  # Ethernet
        if len(frame) < 14:
            return None
        ether_type = struct.unpack_from(">H", frame, 12)[0]
        offset = 14
        while ether_type in (0x8100, 0x88A8, 0x9100):
            if len(frame) < offset + 4:
                return None
            ether_type = struct.unpack_from(">H", frame, offset + 2)[0]
            offset += 4
    elif link_type == 113:  # Linux cooked capture v1
        if len(frame) < 16:
            return None
        ether_type = struct.unpack_from(">H", frame, 14)[0]
        offset = 16
    elif link_type == 276:  # Linux cooked capture v2
        if len(frame) < 20:
            return None
        ether_type = struct.unpack_from(">H", frame, 0)[0]
        offset = 20
    elif link_type in (12, 101):  # Raw IP
        if not frame:
            return None
        ether_type = 0x0800 if frame[0] >> 4 == 4 else 0x86DD
        offset = 0
    elif link_type == 228:  # IPv4
        ether_type, offset = 0x0800, 0
    elif link_type == 229:  # IPv6
        ether_type, offset = 0x86DD, 0
    else:
        raise ValueError(f"Unsupported PCAPNG link type: {link_type}")

    if ether_type == 0x0800:
        if len(frame) < offset + 20 or frame[offset] >> 4 != 4:
            return None
        header_length = (frame[offset] & 0x0F) * 4
        total_length = struct.unpack_from(">H", frame, offset + 2)[0]
        if header_length < 20 or len(frame) < offset + header_length:
            return None
        fragment = struct.unpack_from(">H", frame, offset + 6)[0]
        if fragment & 0x1FFF:
            return None
        protocol = frame[offset + 9]
        source_ip = frame[offset + 12 : offset + 16]
        destination_ip = frame[offset + 16 : offset + 20]
        transport_offset = offset + header_length
        ip_length = total_length
    elif ether_type == 0x86DD:
        if len(frame) < offset + 40 or frame[offset] >> 4 != 6:
            return None
        payload_length = struct.unpack_from(">H", frame, offset + 4)[0]
        protocol = frame[offset + 6]
        source_ip = frame[offset + 8 : offset + 24]
        destination_ip = frame[offset + 24 : offset + 40]
        transport_offset = offset + 40
        ip_length = payload_length + 40
        while protocol in (0, 43, 44, 60, 135, 139, 140):
            if len(frame) < transport_offset + 2:
                return None
            next_protocol = frame[transport_offset]
            if protocol == 44:
                if len(frame) < transport_offset + 8:
                    return None
                fragment = struct.unpack_from(">H", frame, transport_offset + 2)[0]
                if fragment & 0xFFF8:
                    return None
                extension_length = 8
            else:
                extension_length = (frame[transport_offset + 1] + 1) * 8
            if len(frame) < transport_offset + extension_length:
                return None
            transport_offset += extension_length
            protocol = next_protocol
    else:
        return None

    source_port = destination_port = -1
    if protocol in (6, 17) and len(frame) >= transport_offset + 4:
        source_port, destination_port = struct.unpack_from(">HH", frame, transport_offset)
    return Packet(
        timestamp=timestamp,
        source=(source_ip, source_port),
        destination=(destination_ip, destination_port),
        protocol=protocol,
        length=max(0, ip_length),
    )


def _iter_packets(path: Path) -> Iterator[Packet]:
    interfaces: list[Interface] = []
    for block_type, endian, block in _iter_pcapng_blocks(path):
        if block_type == 0x0A0D0D0A:
            interfaces = []
            continue
        body = block[8:-4]
        if block_type == 1:  # Interface Description Block
            if len(body) < 8:
                raise ValueError(f"{path}: malformed interface description block")
            link_type = struct.unpack_from(endian + "H", body, 0)[0]
            resolution = 1e-6
            timestamp_offset = 0
            for code, value in _read_options(body, endian, 8):
                if code == 9 and value:
                    resolution_byte = value[0]
                    exponent = resolution_byte & 0x7F
                    resolution = (
                        2.0**-exponent
                        if resolution_byte & 0x80
                        else 10.0**-exponent
                    )
                elif code == 14 and len(value) == 8:
                    timestamp_offset = struct.unpack(endian + "q", value)[0]
            interfaces.append(Interface(link_type, resolution, timestamp_offset))
            continue

        if block_type == 6:  # Enhanced Packet Block
            if len(body) < 20:
                raise ValueError(f"{path}: malformed enhanced packet block")
            interface_id, ts_high, ts_low, captured_length = struct.unpack_from(
                endian + "IIII", body, 0
            )
            data_offset = 20
        elif block_type == 2:  # Obsolete Packet Block
            if len(body) < 20:
                raise ValueError(f"{path}: malformed packet block")
            interface_id, _, ts_high, ts_low, captured_length, _ = struct.unpack_from(
                endian + "HHIIII", body, 0
            )
            data_offset = 20
        else:
            continue

        if interface_id >= len(interfaces):
            raise ValueError(f"{path}: packet references an unknown interface")
        if data_offset + captured_length > len(body):
            raise ValueError(f"{path}: captured packet exceeds its block")
        interface = interfaces[interface_id]
        timestamp_ticks = (ts_high << 32) | ts_low
        timestamp = (
            timestamp_ticks * interface.timestamp_resolution
            + interface.timestamp_offset
        )
        packet = _packet_from_frame(
            body[data_offset : data_offset + captured_length],
            interface.link_type,
            timestamp,
        )
        if packet is not None:
            yield packet


def _extract_capture(path: Path, label: str, idle_timeout: float) -> list[dict]:
    active_sessions: dict[
        tuple[int, tuple[bytes, int], tuple[bytes, int]], Session
    ] = {}
    rows: list[dict] = []
    for packet in _iter_packets(path):
        endpoints = sorted((packet.source, packet.destination))
        flow_key = (packet.protocol, endpoints[0], endpoints[1])
        session = active_sessions.get(flow_key)
        if (
            session is not None
            and packet.timestamp - session.last_timestamp > idle_timeout
        ):
            rows.append(session.as_row(label))
            session = None
        if session is None:
            active_sessions[flow_key] = Session(packet)
        else:
            session.add(packet)
    rows.extend(session.as_row(label) for session in active_sessions.values())
    return rows


def _split_counts(count: int) -> tuple[int, int, int]:
    ratios = (0.7, 0.1, 0.2)
    raw = [count * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remainder = count - sum(counts)
    order = sorted(range(3), key=lambda index: raw[index] - counts[index], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _split_sessions(
    rows: list[dict], seed: int
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)

    splits = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        samples = by_label[label]
        rng.shuffle(samples)
        train_count, val_count, _ = _split_counts(len(samples))
        splits["train"].extend(samples[:train_count])
        splits["val"].extend(samples[train_count : train_count + val_count])
        splits["test"].extend(samples[train_count + val_count :])
    for samples in splits.values():
        rng.shuffle(samples)
    return splits


def preprocess(dataset_dir: Path, output_dir: Path, seed: int, idle_timeout: float) -> None:
    captures = sorted(dataset_dir.rglob("*.pcap"))
    if not captures:
        raise FileNotFoundError(f"No .pcap files found under {dataset_dir}")

    rows: list[dict] = []
    for capture in captures:
        label = capture.stem.removesuffix("_filtered")
        capture_rows = _extract_capture(capture, label, idle_timeout)
        if not capture_rows:
            raise ValueError(f"{capture}: no supported IP packets were found")
        rows.extend(capture_rows)
        print(f"{capture}: extracted {len(capture_rows)} sessions")

    splits = _split_sessions(rows, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, samples in splits.items():
        output_path = output_dir / f"{split_name}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()
            writer.writerows(samples)
        print(f"{split_name}: {len(samples)} sessions -> {output_path}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extract R1 features and make session-level train/val/test CSVs."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_dir / "dataset",
        help="Folder containing client subfolders and PCAPNG captures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "processed_r1",
        help="Destination folder for train.csv, val.csv, and test.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=60.0,
        help="Seconds of inactivity after which a flow becomes a new session.",
    )
    args = parser.parse_args()
    if args.idle_timeout <= 0:
        parser.error("--idle-timeout must be greater than zero")
    preprocess(args.dataset_dir, args.output_dir, args.seed, args.idle_timeout)


if __name__ == "__main__":
    main()
