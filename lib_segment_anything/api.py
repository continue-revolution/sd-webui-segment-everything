import os
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any, Optional, List
import gradio as gr
from PIL import Image
import numpy as np

from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from lib_segment_anything.sam import (sam_predict, dino_predict, update_mask,
    cnet_seg, categorical_mask, sam_model_list)


def decode_to_pil(image):
    if os.path.exists(image):
        return Image.open(image)
    elif type(image) is str:
        return decode_base64_to_image(image)
    elif type(image) is Image.Image:
        return image
    elif type(image) is np.ndarray:
        return Image.fromarray(image)
    else:
        Exception("Not an image")


def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return encode_pil_to_base64(image).decode()
    elif type(image) is np.ndarray:
        pil = Image.fromarray(image)
        return encode_pil_to_base64(pil).decode()
    else:
        Exception("Invalid type")


def sam_api(_: gr.Blocks, app: FastAPI):    
    @app.get("/sam/heartbeat")
    async def heartbeat():
        return {            
            "msg": "Success!"
        }

    @app.get("/sam/sam-model", description='Query available SAM model')
    async def api_sam_model() -> List[str]:
        return sam_model_list

    class SamPredictRequest(BaseModel):
        sam_model_name: str = "sam_vit_h_4b8939.pth"
        input_image: str
        sam_positive_points: List[List[float]] = []
        sam_negative_points: List[List[float]] = []
        dino_enabled: bool = False
        dino_model_name: Optional[str] = "GroundingDINO_SwinT_OGC (694MB)"
        dino_text_prompt: Optional[str] = None
        dino_box_threshold: Optional[float] = 0.3
        dino_preview_checkbox: bool = False
        dino_preview_boxes_selection: Optional[List[int]] = None

    @app.post("/sam/sam-predict")
    async def api_sam_predict(payload: SamPredictRequest = Body(...)) -> Any:
        print(f"SAM API /sam/sam-predict received request")
        payload.input_image = decode_to_pil(payload.input_image).convert('RGBA')
        sam_output_mask_gallery, sam_message = sam_predict(
            payload.sam_model_name,
            payload.input_image,
            payload.sam_positive_points,
            payload.sam_negative_points,
            payload.dino_enabled,
            payload.dino_model_name,
            payload.dino_text_prompt,
            payload.dino_box_threshold,
            payload.dino_preview_checkbox,
            payload.dino_preview_boxes_selection)
        print(f"SAM API /sam/sam-predict finished with message: {sam_message}")
        result = {
            "msg": sam_message,
        }
        if len(sam_output_mask_gallery) == 9:
            result["blended_images"] = list(map(encode_to_base64, sam_output_mask_gallery[:3]))
            result["masks"] = list(map(encode_to_base64, sam_output_mask_gallery[3:6]))
            result["masked_images"] = list(map(encode_to_base64, sam_output_mask_gallery[6:]))
        return result

    class DINOPredictRequest(BaseModel):
        input_image: str
        dino_model_name: str = "GroundingDINO_SwinT_OGC (694MB)"
        text_prompt: str
        box_threshold: float = 0.3

    @app.post("/sam/dino-predict")
    async def api_dino_predict(payload: DINOPredictRequest = Body(...)) -> Any:
        print(f"SAM API /sam/dino-predict received request")
        payload.input_image = decode_to_pil(payload.input_image)
        dino_output_img, _, dino_msg = dino_predict(
            payload.input_image,
            payload.dino_model_name,
            payload.text_prompt,
            payload.box_threshold)
        if "value" in dino_msg:
            dino_msg = dino_msg["value"]
        else:
            dino_msg = "Done"
        print(f"SAM API /sam/dino-predict finished with message: {dino_msg}")
        return {
            "msg": dino_msg,
            "image_with_box": encode_to_base64(dino_output_img) if dino_output_img is not None else None,
        }

    class DilateMaskRequest(BaseModel):
        input_image: str
        mask: str
        dilate_amount: int = 10

    @app.post("/sam/dilate-mask")
    async def api_dilate_mask(payload: DilateMaskRequest = Body(...)) -> Any:
        print(f"SAM API /sam/dilate-mask received request")
        payload.input_image = decode_to_pil(payload.input_image).convert("RGBA")
        payload.mask = decode_to_pil(payload.mask)
        dilate_result = list(map(encode_to_base64, update_mask(payload.mask, 0, payload.dilate_amount, payload.input_image)))
        print(f"SAM API /sam/dilate-mask finished")
        return {"blended_image": dilate_result[0], "mask": dilate_result[1], "masked_image": dilate_result[2]}

    
    class AutoSAMConfig(BaseModel):
        points_per_side: Optional[int] = 32
        points_per_batch: int = 64
        pred_iou_thresh: float = 0.88
        stability_score_thresh: float = 0.95
        stability_score_offset: float = 1.0
        box_nms_thresh: float = 0.7
        crop_n_layers: int = 0
        crop_nms_thresh: float = 0.7
        crop_overlap_ratio: float = 512 / 1500
        crop_n_points_downscale_factor: int = 1
        min_mask_region_area: int = 0

    class ControlNetSegRequest(BaseModel):
        sam_model_name: str = "sam_vit_h_4b8939.pth"
        input_image: str
        processor: str = "seg_ofade20k"
        processor_res: int = 512
        pixel_perfect: bool = False
        resize_mode: Optional[int] = 1 # 0: just resize, 1: crop and resize, 2: resize and fill
        target_W: Optional[int] = None
        target_H: Optional[int] = None

    @app.post("/sam/controlnet-seg")
    async def api_controlnet_seg(payload: ControlNetSegRequest = Body(...),
                                 autosam_conf: AutoSAMConfig = Body(...)) -> Any:
        print(f"SAM API /sam/controlnet-seg received request")
        payload.input_image = decode_to_pil(payload.input_image)
        cnet_seg_img, cnet_seg_msg = cnet_seg(
            payload.sam_model_name,
            payload.input_image,
            payload.processor,
            payload.processor_res,
            payload.pixel_perfect,
            payload.resize_mode,
            payload.target_W,
            payload.target_H,
            autosam_conf.points_per_side,
            autosam_conf.points_per_batch,
            autosam_conf.pred_iou_thresh,
            autosam_conf.stability_score_thresh,
            autosam_conf.stability_score_offset,
            autosam_conf.box_nms_thresh,
            autosam_conf.crop_n_layers,
            autosam_conf.crop_nms_thresh,
            autosam_conf.crop_overlap_ratio,
            autosam_conf.crop_n_points_downscale_factor,
            autosam_conf.min_mask_region_area)
        cnet_seg_img = list(map(encode_to_base64, cnet_seg_img))
        print(f"SAM API /sam/controlnet-seg finished with message {cnet_seg_msg}")
        result = {
            "msg": cnet_seg_msg,
        }
        if len(cnet_seg_img) == 3:
            result["blended_images"]        = cnet_seg_img[0]
            result["random_seg"]   = cnet_seg_img[1]
            result["edit_anything_control"] = cnet_seg_img[2]
        elif len(cnet_seg_img) == 4:
            result["sem_presam"]        = cnet_seg_img[0]
            result["sem_postsam"]       = cnet_seg_img[1]
            result["blended_presam"]    = cnet_seg_img[2]
            result["blended_postsam"]   = cnet_seg_img[3]
        return result
    
    class CategoryMaskRequest(BaseModel):
        sam_model_name: str = "sam_vit_h_4b8939.pth"
        processor: str = "seg_ofade20k"
        processor_res: int = 512
        pixel_perfect: bool = False
        resize_mode: Optional[int] = 1
        target_W: Optional[int] = None
        target_H: Optional[int] = None
        category: str
        input_image: str
    
    @app.post("/sam/category-mask")
    async def api_category_mask(payload: CategoryMaskRequest = Body(...),
                                autosam_conf: AutoSAMConfig = Body(...)) -> Any:
        print(f"SAM API /sam/category-mask received request")
        payload.input_image = decode_to_pil(payload.input_image)
        category_mask_img, category_mask_msg, resized_input_img = categorical_mask(
            payload.sam_model_name,
            payload.processor,
            payload.processor_res,
            payload.pixel_perfect,
            payload.resize_mode,
            payload.target_W,
            payload.target_H,
            payload.category,
            payload.input_image,
            autosam_conf.points_per_side,
            autosam_conf.points_per_batch,
            autosam_conf.pred_iou_thresh,
            autosam_conf.stability_score_thresh,
            autosam_conf.stability_score_offset,
            autosam_conf.box_nms_thresh,
            autosam_conf.crop_n_layers,
            autosam_conf.crop_nms_thresh,
            autosam_conf.crop_overlap_ratio,
            autosam_conf.crop_n_points_downscale_factor,
            autosam_conf.min_mask_region_area)
        category_mask_img = list(map(encode_to_base64, category_mask_img))
        print(f"SAM API /sam/category-mask finished with message {category_mask_msg}")
        result = {
            "msg": category_mask_msg,
        }
        if len(category_mask_img) == 3:
            result["blended_image"] = category_mask_img[0]
            result["mask"]          = category_mask_img[1]
            result["masked_image"]  = category_mask_img[2]
        if resized_input_img is not None:
            result["resized_input"] = encode_to_base64(resized_input_img)
        return result

