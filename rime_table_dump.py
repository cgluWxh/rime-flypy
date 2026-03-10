#!/usr/bin/env python3
"""
RIME table.bin → dict.yaml reverser
Based on librime table.h / mapped_file structures

Key structures (all little-endian):
  Metadata   @ file base
  OffsetPtr  = int32_t offset from &self (0 means null)
  Array<T>   = { int32_t size; T data[size]; }  (4-byte aligned)
  List<T>    = same layout as Array in practice (linked via OffsetPtr)
  String     = OffsetPtr<char>  (points to null-terminated string in file)
  StringId   = int32_t          (marisa trie key id, used when string_table present)
  StringType = union { String str; StringId str_id; }  (int32_t either way)
  Entry      = { StringType text; float weight; }  = 8 bytes
  LongEntry  = { Code extra_code; Entry entry; }
  Code/List<SyllableId> = { int32_t size; OffsetPtr<int32_t[]> data_offset; }

HeadIndex  = Array<HeadIndexNode>
  HeadIndexNode = { List<Entry> entries; OffsetPtr<PhraseIndex> next_level; }
  — indexed by syllable_id directly (position = syllable_id)

TrunkIndex = Array<TrunkIndexNode>
  TrunkIndexNode = { int32_t key(SyllableId); List<Entry> entries; OffsetPtr<PhraseIndex> next_level; }

TailIndex  = Array<LongEntry>
  LongEntry = { Code extra_code; Entry entry; }
"""

import struct
import sys
import os
import io
import marisa_trie
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Low-level memory view helpers
# ---------------------------------------------------------------------------

class Region:
    """Wraps a bytes-like object with offset-aware reads."""
    def __init__(self, data: bytes):
        self.data = data

    def u8(self, off): return struct.unpack_from('<B', self.data, off)[0]
    def i32(self, off): return struct.unpack_from('<i', self.data, off)[0]
    def u32(self, off): return struct.unpack_from('<I', self.data, off)[0]
    def f32(self, off): return struct.unpack_from('<f', self.data, off)[0]

    def deref_offset_ptr(self, ptr_addr) -> Optional[int]:
        """OffsetPtr<T>: stored as int32_t offset from its own address."""
        off = self.i32(ptr_addr)
        if off == 0:
            return None
        return ptr_addr + off

    def cstr(self, addr) -> str:
        end = self.data.index(b'\x00', addr)
        return self.data[addr:end].decode('utf-8', errors='replace')

    def array_size(self, addr) -> int:
        return self.i32(addr)

    def array_data_start(self, addr) -> int:
        return addr + 4  # skip int32_t size field


# ---------------------------------------------------------------------------
# Metadata layout  (offsets from file start)
# ---------------------------------------------------------------------------
# struct Metadata {
#   char format[32];           @ 0
#   uint32_t dict_file_checksum; @ 32
#   uint32_t num_syllables;    @ 36
#   uint32_t num_entries;      @ 40
#   OffsetPtr<Syllabary> syllabary; @ 44
#   OffsetPtr<Index>     index;     @ 48
#   int32_t reserved_1;        @ 52
#   int32_t reserved_2;        @ 56
#   OffsetPtr<char> string_table;  @ 60
#   uint32_t string_table_size;    @ 64
# };  total = 68 bytes

META_FORMAT      = 0
META_CHECKSUM    = 32
META_NUM_SYLL    = 36
META_NUM_ENTRIES = 40
META_SYLLABARY   = 44
META_INDEX       = 48
META_RESERVED1   = 52
META_RESERVED2   = 56
META_STRTABLE    = 60
META_STRTABLE_SZ = 64


# ---------------------------------------------------------------------------
# Entry sizes
# ---------------------------------------------------------------------------
# StringType = int32_t (4 bytes)
# Weight = float (4 bytes)
# Entry = StringType + Weight = 8 bytes
ENTRY_SIZE = 8

# HeadIndexNode = List<Entry> + OffsetPtr<PhraseIndex>
#   List<Entry>: { int32_t size; OffsetPtr<Entry[]> data; } = 8 bytes
#   OffsetPtr: 4 bytes
# Total: 12 bytes
HEAD_NODE_SIZE = 12

# TrunkIndexNode = int32_t key + List<Entry> + OffsetPtr<PhraseIndex>
#   = 4 + 8 + 4 = 16 bytes
TRUNK_NODE_SIZE = 16

# Code = List<SyllableId> = { int32_t size; OffsetPtr<int32_t[]> data; } = 8 bytes
CODE_SIZE = 8

# LongEntry = Code + Entry = 8 + 8 = 16 bytes
LONG_ENTRY_SIZE = 16


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

@dataclass
class DictEntry:
    text: str
    weight: float
    code: list  # list of syllable strings


class RimeTableDumper:
    def __init__(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.raw = f.read()
        self.r = Region(self.raw)
        self.syllabary = []       # index → syllable string
        self.trie = None          # marisa trie for v2 string table
        self.has_string_table = False
        self.entries = []         # collected DictEntry list

    # ------------------------------------------------------------------
    def load(self):
        r = self.r

        fmt = self.raw[META_FORMAT:META_FORMAT+32].rstrip(b'\x00').decode()
        print(f"[*] Format: {fmt}")

        num_syll    = r.u32(META_NUM_SYLL)
        num_entries = r.u32(META_NUM_ENTRIES)
        print(f"[*] Syllables: {num_syll}, Entries: {num_entries}")

        # String table (v2)
        st_ptr_addr = META_STRTABLE
        st_addr = r.deref_offset_ptr(st_ptr_addr)
        st_size = r.u32(META_STRTABLE_SZ)

        if st_addr and st_size > 0:
            self.has_string_table = True
            st_data = self.raw[st_addr: st_addr + st_size]
            self.trie = marisa_trie.Trie()
            self.trie.frombytes(st_data)
            print(f"[*] String table (marisa trie) loaded, size={st_size}, keys={len(self.trie)}")
        else:
            print("[*] No string table (v1, inline strings)")

        # Syllabary
        syll_ptr_addr = META_SYLLABARY
        syll_addr = r.deref_offset_ptr(syll_ptr_addr)
        if syll_addr is None:
            raise RuntimeError("syllabary pointer is null")
        self._load_syllabary(syll_addr, num_syll)

        # Index
        idx_ptr_addr = META_INDEX
        idx_addr = r.deref_offset_ptr(idx_ptr_addr)
        if idx_addr is None:
            raise RuntimeError("index pointer is null")

        print(f"[*] Walking index @ 0x{idx_addr:x} ...")
        self._walk_head_index(idx_addr)
        print(f"[*] Total entries collected: {len(self.entries)}")

    # ------------------------------------------------------------------
    def _load_syllabary(self, addr, num_syll):
        """Array<StringType>: size + StringType[size], each StringType=int32_t."""
        r = self.r
        size = r.array_size(addr)
        data_start = r.array_data_start(addr)
        print(f"[*] Syllabary array size={size} @ 0x{addr:x}")
        for i in range(size):
            st_addr = data_start + i * 4  # StringType = 4 bytes
            self.syllabary.append(self._read_string_type(st_addr))
        if len(self.syllabary) >= 3:
            print(f"    first few: {self.syllabary[:5]}")

    # ------------------------------------------------------------------
    def _read_string_type(self, addr) -> str:
        """Read StringType (union of String OffsetPtr or StringId)."""
        r = self.r
        val = r.i32(addr)
        if self.has_string_table:
            # val is StringId (marisa key id)
            if val < 0 or val >= len(self.trie):
                return f"<bad_id:{val}>"
            try:
                return self.trie.restore_key(val)
            except Exception:
                return f"<bad_id:{val}>"
        else:
            # val is OffsetPtr<char>: offset from addr
            if val == 0:
                return ""
            char_addr = addr + val
            return r.cstr(char_addr)

    # ------------------------------------------------------------------
    def _read_entries_list(self, list_addr) -> list:
        """
        List<Entry> = { int32_t size; OffsetPtr<Entry[]> data_ptr }
        Returns list of (text:str, weight:float)
        """
        r = self.r
        size = r.i32(list_addr)
        if size <= 0 or size > 100000:
            return []
        data_ptr_addr = list_addr + 4
        data_addr = r.deref_offset_ptr(data_ptr_addr)
        if data_addr is None:
            return []
        result = []
        for i in range(size):
            e_addr = data_addr + i * ENTRY_SIZE
            text = self._read_string_type(e_addr)
            weight = r.f32(e_addr + 4)
            result.append((text, weight))
        return result

    # ------------------------------------------------------------------
    def _walk_head_index(self, head_addr):
        """
        HeadIndex = Array<HeadIndexNode>
        HeadIndexNode @ position i corresponds to syllable_id = i
        """
        r = self.r
        size = r.array_size(head_addr)
        data_start = r.array_data_start(head_addr)
        print(f"    HeadIndex size={size}")

        for syll_id in range(size):
            node_addr = data_start + syll_id * HEAD_NODE_SIZE
            # List<Entry> entries @ +0 (8 bytes)
            entries = self._read_entries_list(node_addr)
            code_str = [self.syllabary[syll_id]] if syll_id < len(self.syllabary) else [f"?{syll_id}"]
            for text, weight in entries:
                self.entries.append(DictEntry(text=text, weight=weight, code=code_str[:]))

            # OffsetPtr<PhraseIndex> next_level @ +8
            next_ptr_addr = node_addr + 8
            next_addr = r.deref_offset_ptr(next_ptr_addr)
            if next_addr:
                self._walk_trunk_index(next_addr, code_str, level=2)

    # ------------------------------------------------------------------
    def _walk_trunk_index(self, trunk_addr, prefix_code: list, level: int):
        """
        TrunkIndex = Array<TrunkIndexNode>
        TrunkIndexNode = { int32_t key; List<Entry> entries(8); OffsetPtr<PhraseIndex> next(4); }
        """
        r = self.r
        size = r.array_size(trunk_addr)
        if size <= 0 or size > 500000:
            return
        data_start = r.array_data_start(trunk_addr)

        for i in range(size):
            node_addr = data_start + i * TRUNK_NODE_SIZE
            key = r.i32(node_addr)  # SyllableId
            syll = self.syllabary[key] if 0 <= key < len(self.syllabary) else f"?{key}"
            code_str = prefix_code + [syll]

            entries = self._read_entries_list(node_addr + 4)
            for text, weight in entries:
                self.entries.append(DictEntry(text=text, weight=weight, code=code_str[:]))

            next_ptr_addr = node_addr + 4 + 8  # skip key(4) + List<Entry>(8)
            next_addr = r.deref_offset_ptr(next_ptr_addr)
            if next_addr and level < 4:
                self._walk_trunk_or_tail(next_addr, code_str, level + 1)

    # ------------------------------------------------------------------
    def _walk_trunk_or_tail(self, addr, prefix_code: list, level: int):
        """At level 4 it's a TailIndex, otherwise TrunkIndex."""
        if level >= 4:
            self._walk_tail_index(addr, prefix_code)
        else:
            self._walk_trunk_index(addr, prefix_code, level)

    # ------------------------------------------------------------------
    def _walk_tail_index(self, tail_addr, prefix_code: list):
        """
        TailIndex = Array<LongEntry>
        LongEntry = { Code extra_code(8); Entry entry(8); } = 16 bytes
        Code = List<SyllableId> = { int32_t size; OffsetPtr<SyllableId[]> data; }
        """
        r = self.r
        size = r.array_size(tail_addr)
        if size <= 0 or size > 100000:
            return
        data_start = r.array_data_start(tail_addr)

        for i in range(size):
            le_addr = data_start + i * LONG_ENTRY_SIZE
            # Code extra_code: { int32_t size; OffsetPtr<SyllableId[]> data; }
            extra_size = r.i32(le_addr)
            extra_syls = []
            if 0 < extra_size < 20:
                data_ptr_addr = le_addr + 4
                data_addr = r.deref_offset_ptr(data_ptr_addr)
                if data_addr:
                    for j in range(extra_size):
                        sid = r.i32(data_addr + j * 4)
                        extra_syls.append(
                            self.syllabary[sid] if 0 <= sid < len(self.syllabary) else f"?{sid}"
                        )
            # Entry @ +8
            entry_addr = le_addr + 8
            text = self._read_string_type(entry_addr)
            weight = r.f32(entry_addr + 4)
            full_code = prefix_code + extra_syls
            self.entries.append(DictEntry(text=text, weight=weight, code=full_code))

    # ------------------------------------------------------------------
    def dump_yaml(self, out_path: str):
        """Write dict.yaml compatible output."""
        # sort by code then weight desc
        self.entries.sort(key=lambda e: (''.join(e.code), -e.weight))

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("# Recovered by rime_table_dump.py\n")
            f.write("---\n")
            f.write("name: recovered\n")
            f.write("version: \"1.0\"\n")
            f.write("sort: by_weight\n")
            f.write("use_preset_vocabulary: false\n")
            f.write("...\n\n")
            for e in self.entries:
                code = ' '.join(e.code)
                # weight: if it looks like an integer store as int
                if e.weight == int(e.weight):
                    w = str(int(e.weight))
                else:
                    w = f"{e.weight:.6g}"
                f.write(f"{e.text}\t{code}\t{w}\n")
        print(f"[*] Wrote {len(self.entries)} entries to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 rime_table_dump.py <table.bin> [output.dict.yaml]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path.replace('.bin', '.dict.yaml')
    out_path = out_path if out_path.endswith('.yaml') else out_path + '.dict.yaml'

    print(f"[*] Input : {in_path}")
    print(f"[*] Output: {out_path}")

    dumper = RimeTableDumper(in_path)
    dumper.load()
    dumper.dump_yaml(out_path)
    print("[*] Done.")

if __name__ == '__main__':
    main()