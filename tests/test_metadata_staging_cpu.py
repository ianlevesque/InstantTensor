"""Execute exact Python constructor/budget/open checks; no Torch/CUDA import."""
import ast
from dataclasses import make_dataclass,replace
from enum import IntEnum
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import unittest
import warnings

class Backend(IntEnum):
    MMAP=0;CUFILE=1;URING=2;AIO=3;URING_BUFFERED=4;AIO_BUFFERED=5

class MetadataStagingTest(unittest.TestCase):
    def setUp(self):
        self.opens=[]
        config=make_dataclass('Config',[(n,object) for n in ['buffer_size','chunk_size','concurrency','io_depth','max_free_mem_usage','backend']],frozen=True)
        self.ns=dict(os=os,sys=sys,time=time,warnings=warnings,math=math,replace=replace,Backend=Backend,
            _resolve_open_config=lambda *args:config(*args),_compile_name_matcher=lambda names,mode:lambda name:names is None or name in names,
            torch=NS(device=lambda value:NS(type='cuda',index=0),cuda=NS(get_device_properties=lambda dev:NS(is_integrated=True))),
            dist=NS(get_world_size=lambda group:16,get_rank=lambda group:0),cpu_count=lambda:16,
            parse_backend_candidates=lambda value:[value],select_backend=lambda value:value[0],file_in_memory=lambda path:False,
            MAX_CHUNK_SIZE=1<<30,MAX_IO_DEPTH=1<<20,debug_log=lambda *args:None,_host_available_bytes=lambda:1<<30,
            required_buffer_size_for_io=lambda chunk,depth,world:((chunk+4095)//4096)*4096*depth*world,
            recommended_buffer_size_for_tensors=lambda sizes:2*max(sizes,default=0),
            _C=NS(open=lambda *args:self.opens.append(args) or 1))
        tree=ast.parse((Path(__file__).parents[1]/'instanttensor/_impl.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='safe_open')
        exec('from __future__ import annotations\n'+ast.unparse(cls),self.ns)
        self.cls=self.ns['safe_open']
        self.cls._read_metadata=lambda obj:[(None,{'small':{'shape':[4],'dtype':'U8','data_offsets':[0,16]}},128)]

    def construct(self,**kwargs):
        return self.cls(['file'],'pt','cuda:0',buffer_size=1,chunk_size=4096,io_depth=3,backend=Backend.URING,load_now=False,**kwargs)

    def test_default_tiny_ring_rejected(self):
        with self.assertRaisesRegex(ValueError,'too small'):self.construct()
        self.assertEqual(self.opens,[])

    def test_defer_rejects_immediate_open_and_borrowed_views(self):
        for params in [dict(load_now=True),dict(load_now=False,copy=False)]:
            with self.assertRaisesRegex(ValueError,'load_now=False and copy=True'):
                self.cls(['file'],'pt','cuda:0',defer_layout_validation=True,**params)
        self.assertEqual(self.opens,[])

    def test_unfinalized_and_empty_never_open(self):
        obj=self.construct(defer_layout_validation=True)
        with self.assertRaisesRegex(RuntimeError,'not been finalized'):obj._open()
        obj.tensor_sizes=[]
        with self.assertRaisesRegex(ValueError,'Empty'):obj._finalize_buffer_size(1)
        with self.assertRaisesRegex(RuntimeError,'not been finalized'):obj._open()
        self.assertEqual(self.opens,[])

    def test_selected_minimum_budget_and_open(self):
        obj=self.construct(defer_layout_validation=True)
        with self.assertRaisesRegex(ValueError,'I/O minimum'):obj._finalize_buffer_size(12287)
        obj._finalize_buffer_size(12288)
        self.assertEqual(obj.buffer_size,12288)
        obj._open()
        self.assertEqual(len(self.opens),1)
        self.assertTrue(obj._layout_finalized)

    def test_device_host_budget_and_collective_minimum(self):
        obj=self.construct(defer_layout_validation=True)
        obj._device_memory_budget=12288
        with self.assertRaisesRegex(RuntimeError,'staging exceeds'):obj._finalize_buffer_size(12288)
        self.assertFalse(obj._layout_finalized)
        obj._device_memory_budget=1<<30;obj.world_size=16
        with self.assertRaisesRegex(ValueError,'I/O minimum'):obj._finalize_buffer_size(12288)
        obj._finalize_buffer_size(196608)
        self.assertTrue(obj._layout_finalized)

if __name__=='__main__':unittest.main()
