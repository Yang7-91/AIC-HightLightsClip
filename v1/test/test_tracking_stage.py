import ast
from contextlib import nullcontext
import importlib.util
import io
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from tracking_stage import (
    _legal_crop_from_state,
    fill_missing_boxes,
    identify_bad_track_frames,
    plan_smoothed_crops,
    subject_box_to_crop,
    SAM2VideoTracker,
)


class TrackingGeometryTests(unittest.TestCase):
    def test_vertical_max_width_is_floored(self):
        box = _legal_crop_from_state(960, 540, 9999, (1920, 1080), (9, 16))
        self.assertEqual(box[2], 607)
        self.assertLessEqual(box[1] + box[2] * 16 / 9, 1080)

    def test_horizontal_and_vertical_always_legal(self):
        for size in ((1920,1080),(1080,1920),(721,1281),(640,360)):
            for ratio in ((16,9),(9,16)):
                for cx,cy,w in ((-100,-100,1),(0,0,10000),(size[0],size[1],333)):
                    with self.subTest(size=size,ratio=ratio,cx=cx,cy=cy,w=w):
                        x,y,width,height=_legal_crop_from_state(cx,cy,w,size,ratio)
                        self.assertGreater(width,0)
                        self.assertGreaterEqual(x,0); self.assertGreaterEqual(y,0)
                        self.assertLessEqual(x+width,size[0])
                        self.assertLessEqual(y+height,size[1]+1e-9)

    def test_scale_adapts_to_subject(self):
        small=subject_box_to_crop([900,450,1020,650],(1920,1080),(16,9))
        large=subject_box_to_crop([100,50,1800,1000],(1920,1080),(16,9))
        self.assertLess(small[2],large[2])

    def test_smoothing_covers_every_frame_and_is_legal(self):
        subjects={f:[100+f*10,100,300+f*10,500] for f in range(11)}
        crops=plan_smoothed_crops(subjects,0,10,(1280,720),(9,16))
        self.assertEqual(set(crops),set(range(11)))
        for x,y,w,h in crops.values():
            self.assertLessEqual(x+w,1280); self.assertLessEqual(y+h,720+1e-9)

    def test_missing_fill_is_shot_local(self):
        result=fill_missing_boxes({2:[2,2,12,12]},0,4,[0,0,20,20])
        self.assertEqual(result[0],[2.0,2.0,12.0,12.0])
        self.assertEqual(result[4],[2.0,2.0,12.0,12.0])
        fallback=fill_missing_boxes({},0,1,[0,0,20,20])
        self.assertEqual(fallback[0],[0.0,0.0,20.0,20.0])

    def test_bad_track_detection(self):
        boxes={0:[10,10,30,30],1:None,2:[0,0,100,100],3:[90,90,91,91]}
        bad=identify_bad_track_frames(boxes,0,3,(100,100),max_area_ratio=.7)
        self.assertIn(1,bad); self.assertIn(2,bad); self.assertIn(3,bad)
        self.assertNotIn(0,bad)

    def test_v1_source_contains_new_backend(self):
        path=Path(__file__).with_name('v1_qwen.py')
        tree=ast.parse(path.read_text(encoding='utf-8'))
        functions={node.name for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
        self.assertIn('track_segment',functions)
        self.assertIn('predict_subject_box',functions)

    def test_parse_subject_box_from_v1(self):
        path=Path(__file__).with_name('v1_qwen.py')
        tree=ast.parse(path.read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_subject_box')
        namespace={'json':__import__('json'),'re':__import__('re'),'math':math}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        parse=namespace['parse_subject_box']
        self.assertEqual(parse('{"subject_box":[0,0,1000,1000]}',1920,1080),[0.0,0.0,1920.0,1080.0])
        self.assertEqual(parse('{"subject_box":[0.25,0.25,0.75,0.75]}',100,200),[25.0,50.0,75.0,150.0])
        self.assertIsNone(parse('{"subject_box":[500,500,400,600]}',1920,1080))

    def test_sam_adapter_frame_mapping_with_fake_predictor(self):
        class Predictor:
            def __init__(self): self.prompt=None
            def init_state(self,video_path): return {'path':video_path}
            def reset_state(self,state): pass
            def add_new_points_or_box(self,**kwargs):
                self.prompt=kwargs
                return 0,[1],np.ones((1,2,2),dtype=np.float32)
            def propagate_in_video(self,state):
                for frame in range(3):
                    yield frame,[1],np.ones((1,2,2),dtype=np.float32)
        tracker=object.__new__(SAM2VideoTracker)
        tracker.predictor=Predictor(); tracker.device='cpu'; tracker.amp_dtype='none'
        tracker._contexts=lambda:nullcontext()
        with patch('tracking_stage._decode_range_to_jpegs'), \
             patch('tracking_stage.largest_component_box',return_value=[1,2,3,4]), \
             patch.dict('sys.modules',{'cv2':type('CV2',(),{'INTER_NEAREST':0,'resize':staticmethod(lambda a,s,interpolation:a)})}):
            result=tracker.track_range('fake.mp4',10,12,[1,2,3,4],(2,2))
        self.assertEqual(result,{10:[1,2,3,4],11:[1,2,3,4],12:[1,2,3,4]})
        self.assertEqual(tracker.predictor.prompt['frame_idx'],0)

    def test_stage2_orchestration_with_fake_qwen_and_sam(self):
        cv2=ModuleType('cv2')
        cv2.COLOR_BGR2RGB=1
        cv2.cvtColor=lambda frame,code:frame
        openai=ModuleType('openai')
        openai.OpenAI=object
        path=Path(__file__).with_name('v1_qwen.py')
        spec=importlib.util.spec_from_file_location('v1_qwen_under_test',path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'cv2':cv2,'openai':openai}):
            spec.loader.exec_module(module)

        class Qwen:
            def predict_subject_box(self,image,ratio,tokens):
                return {'content':'{"subject_box":[250,200,750,900]}','reasoning':None}
        class Tracker:
            def track_range(self,video,start,end,box,size):
                return {f:[480+10*(f-start),216,1440+10*(f-start),972]
                        for f in range(start,end+1)}
        args=SimpleNamespace(
            scene_threshold=27.0,scene_min_frames=15,crop_scales='0.55,0.70,0.85,1.0',
            margin_left=.2,margin_right=.2,margin_top=.15,margin_bottom=.3,
            subject_max_tokens=256,min_mask_area_ratio=.0005,max_mask_area_ratio=.7,
            max_area_change=4.0,max_center_jump=.2,tracking_reinit=1,
            center_alpha=.25,width_alpha=.15,tracking_fallback='error',crop_stride=15)
        frame=np.zeros((1080,1920,3),dtype=np.uint8)
        with patch.object(module,'detect_shots',return_value=[(10,12)]), \
             patch.object(module,'extract_frames',side_effect=lambda video,ids:{ids[0]:frame}):
            log=io.StringIO()
            result=module.track_segment(Qwen(),Tracker(),'fake.mp4',(10,12),(9,16),
                                        1920,1080,log,'0',args,607,1080)
        self.assertEqual([item['frame'] for item in result],[10,11,12])
        for item in result:
            x,y,w=item['bboxes']
            self.assertLessEqual(x+w,1920)
            self.assertLessEqual(y+w*16/9,1080)
        self.assertIn('tracking_anchor',log.getvalue())
        self.assertIn('tracking_summary',log.getvalue())


if __name__ == '__main__':
    unittest.main()
