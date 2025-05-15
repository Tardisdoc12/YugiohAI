################################################################################
# filename: Reader.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 15/05,2025
################################################################################

import struct
import lzma

################################################################################

REPLAY_COMPRESSED = 0x1
REPLAY_EXTENDED_HEADER = 0x4
REPLAY_YRP1 = 0x31707279
REPLAY_YRPX = 0x58707279

################################################################################

class ReplayHeader:
    def __init__(self, data):
        self.id, self.version, self.flag, self.datasize = struct.unpack("<IHHI", data)

################################################################################

class ExtendedReplayHeader(ReplayHeader):
    def __init__(self, data):
        super().__init__(data[:12])
        self.props = data[12:17]
        self.header_version = data[17]

################################################################################

def read_replay_file(path):
    with open(path, "rb") as f:
        header_data = f.read(18)  # ExtendedReplayHeader = 18 bytes
        if len(header_data) < 12:
            raise ValueError("Fichier trop court")

        base = ReplayHeader(header_data[:12])
        if base.flag & REPLAY_EXTENDED_HEADER:
            header = ExtendedReplayHeader(header_data)
        else:
            header = base
            header.props = b"\x5d\x00\x00\x80\x00"

        print(
            f"Format: {hex(header.id)} | Flags: {hex(header.flag)} | Version: {header.version}"
        )
        if not (header.id == REPLAY_YRP1 or header.id == REPLAY_YRPX):
            raise ValueError("Fichier invalide")

        remaining = f.read()

        if header.flag & REPLAY_COMPRESSED:
            print("Décompression avec LZMA...")
            filters = [
                {
                    "id": lzma.FILTER_LZMA1,
                    "dict_size": 1 << 24,
                    "lc": header.props[0] % 9,
                    "lp": (header.props[0] // 9) % 5,
                    "pb": (header.props[0] // 45),
                }
            ]

            decompressed = lzma.decompress(
                remaining, format=lzma.FORMAT_RAW, filters=filters
            )
        else:
            decompressed = remaining

        return decompressed  # Tu peux ensuite parser les données avec struct ou autre

################################################################################

replay_data = read_replay_file("dataset/2025-05-13 16-48-12.yrpX")
print(f"Décompressé : {len(replay_data)} octets")

################################################################################
# End of File
################################################################################