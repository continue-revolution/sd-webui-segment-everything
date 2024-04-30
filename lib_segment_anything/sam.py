import gc
import os
import copy
import glob
import numpy as np
from PIL import Image
import torch
import gradio as gr
from collections import OrderedDict
from scipy.ndimage import binary_dilation
from modules import scripts, shared
from modules.ui import gr_show
from modules.ui_components import FormRow
from modules.safe import unsafe_torch_load, load
from modules.devices import device, torch_gc, cpu
from modules.paths import models_path
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry
from scripts.dino import dino_predict_internal, show_boxes, clear_dino_cache, dino_install_issue_text
from scripts.auto import clear_sem_sam_cache, register_auto_sam, semantic_segmentation, sem_sam_garbage_collect, image_layer_internal, categorical_mask_image


refresh_symbol = '\U0001f504'       # 🔄
sam_model_cache = OrderedDict()
scripts_sam_model_dir = os.path.join(scripts.basedir(), "models/sam") 
sd_sam_model_dir = os.path.join(models_path, "sam")
sam_model_dir = sd_sam_model_dir if os.path.exists(sd_sam_model_dir) else scripts_sam_model_dir 
sam_model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']
sam_device = device

def get_sam_model_list():
    return sam_model_list

txt2img_width: gr.Slider = None
txt2img_height: gr.Slider = None
img2img_width: gr.Slider = None
img2img_height: gr.Slider = None


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"
        

def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)


def update_mask(mask_gallery, chosen_mask, dilation_amt, input_image):
    print("Dilation Amount: ", dilation_amt)
    if isinstance(mask_gallery, list):
        mask_image = Image.open(mask_gallery[chosen_mask + 3]['name'])
    else:
        mask_image = mask_gallery
    binary_img = np.array(mask_image.convert('1'))
    if dilation_amt:
        mask_image, binary_img = dilate_mask(binary_img, dilation_amt)
    blended_image = Image.fromarray(show_masks(np.array(input_image), binary_img.astype(np.bool_)[None, ...]))
    matted_image = np.array(input_image)
    matted_image[~binary_img] = np.array([0, 0, 0, 0])
    return [blended_image, mask_image, Image.fromarray(matted_image)]


def load_sam_model(sam_checkpoint):
    model_type = sam_checkpoint.split('.')[0]
    if 'hq' not in model_type and 'mobile' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam_checkpoint_path = os.path.join(sam_model_dir, sam_checkpoint)
    torch.load = unsafe_torch_load
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam.to(device=sam_device)
    sam.eval()
    torch.load = load
    return sam


def clear_sam_cache():
    sam_model_cache.clear()
    gc.collect()
    torch_gc()


def clear_cache():
    clear_sam_cache()
    clear_dino_cache()
    clear_sem_sam_cache()


def garbage_collect(sam):
    if shared.cmd_opts.lowvram:
        sam.to(cpu)
    gc.collect()
    torch_gc()


def refresh_sam_models(*inputs):
    global sam_model_list
    sam_model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
        os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']
    dd = inputs[0]
    if dd in sam_model_list:
        selected = dd
    elif len(sam_model_list) > 0:
        selected = sam_model_list[0]
    else:
        selected = None
    return gr.Dropdown.update(choices=sam_model_list, value=selected)


def init_sam_model(sam_model_name):
    print(f"Initializing SAM to {sam_device}")
    if sam_model_name in sam_model_cache:
        sam = sam_model_cache[sam_model_name]
        if shared.cmd_opts.lowvram or (str(sam_device) not in str(sam.device)):
            sam.to(device=sam_device)
        return sam
    elif sam_model_name in sam_model_list:
        clear_sam_cache()
        sam_model_cache[sam_model_name] = load_sam_model(sam_model_name)
        return sam_model_cache[sam_model_name]
    else:
        raise Exception(
            f"{sam_model_name} not found, please download model to models/sam.")


def dilate_mask(mask, dilation_amt):
    x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
    center = dilation_amt // 2
    dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)
    dilated_binary_img = binary_dilation(mask, dilation_kernel)
    dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)
    return dilated_mask, dilated_binary_img


def create_mask_output(image_np, masks, boxes_filt):
    print("Creating output image")
    mask_images, masks_gallery, matted_images = [], [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))
        blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
        mask_images.append(Image.fromarray(blended_image))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        matted_images.append(Image.fromarray(image_np_copy))
    return mask_images + masks_gallery + matted_images


def create_mask_batch_output(
    input_image_file, dino_batch_dest_dir, 
    image_np, masks, boxes_filt, batch_dilation_amt, 
    dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask):
    print("Creating batch output image")
    filename, ext = os.path.splitext(os.path.basename(input_image_file))
    ext = ".png" # JPEG not compatible with RGBA
    for idx, mask in enumerate(masks):
        blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
        merged_mask = np.any(mask, axis=0)
        if dino_batch_save_background:
            merged_mask = ~merged_mask
        if batch_dilation_amt:
            _, merged_mask = dilate_mask(merged_mask, batch_dilation_amt)
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~merged_mask] = np.array([0, 0, 0, 0])
        if dino_batch_save_image:
            output_image = Image.fromarray(image_np_copy)
            output_image.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_output{ext}"))
        if dino_batch_save_mask:
            output_mask = Image.fromarray(merged_mask)
            output_mask.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_mask{ext}"))
        if dino_batch_save_image_with_mask:
            output_blend = Image.fromarray(blended_image)
            output_blend.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_blend{ext}"))


def sam_predict(sam_model_name, input_image, positive_points, negative_points,
                dino_checkbox, dino_model_name, text_prompt, box_threshold,
                dino_preview_checkbox, dino_preview_boxes_selection):
    print("Start SAM Processing")
    if sam_model_name is None:
        return [], "SAM model not found. Please download SAM model from extension README."
    if input_image is None:
        return [], "SAM requires an input image. Please upload an image first."
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]
    dino_enabled = dino_checkbox and text_prompt is not None
    boxes_filt = None
    sam_predict_result = " done."
    if dino_enabled:
        boxes_filt, install_success = dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold)
        if dino_preview_checkbox is not None and dino_preview_checkbox and dino_preview_boxes_selection is not None:
            valid_indices = [int(i) for i in dino_preview_boxes_selection if int(i) < boxes_filt.shape[0]]
            boxes_filt = boxes_filt[valid_indices]
    sam = init_sam_model(sam_model_name)
    print(f"Running SAM Inference {image_np_rgb.shape}")
    predictor = SamPredictorHQ(sam, 'hq' in sam_model_name)
    predictor.set_image(image_np_rgb)
    if dino_enabled and boxes_filt.shape[0] > 1:
        sam_predict_status = f"SAM inference with {boxes_filt.shape[0]} boxes, point prompts discarded"
        print(sam_predict_status)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(sam_device),
            multimask_output=True)
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    else:
        num_box = 0 if boxes_filt is None else boxes_filt.shape[0]
        num_points = len(positive_points) + len(negative_points)
        if num_box == 0 and num_points == 0:
            garbage_collect(sam)
            if dino_enabled and dino_preview_checkbox and num_box == 0:
                return [], "It seems that you are using a high box threshold with no point prompts. Please lower your box threshold and re-try."
            return [], "You neither added point prompts nor enabled GroundingDINO. Segmentation cannot be generated."
        sam_predict_status = f"SAM inference with {num_box} box, {len(positive_points)} positive prompts, {len(negative_points)} negative prompts"
        print(sam_predict_status)
        point_coords = np.array(positive_points + negative_points)
        point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
        box = copy.deepcopy(boxes_filt[0].numpy()) if boxes_filt is not None and boxes_filt.shape[0] > 0 else None
        masks, _, _ = predictor.predict(
            point_coords=point_coords if len(point_coords) > 0 else None,
            point_labels=point_labels if len(point_coords) > 0 else None,
            box=box,
            multimask_output=True)
        masks = masks[:, None, ...]
    garbage_collect(sam)
    return create_mask_output(image_np, masks, boxes_filt), sam_predict_status + sam_predict_result + (f" However, GroundingDINO installment has failed. Your process automatically fall back to local groundingdino. Check your terminal for more detail and {dino_install_issue_text}." if (dino_enabled and not install_success) else "")


def dino_predict(input_image, dino_model_name, text_prompt, box_threshold):
    if input_image is None:
        return None, gr.update(), gr.update(visible=True, value=f"GroundingDINO requires input image.")
    if text_prompt is None or text_prompt == "":
        return None, gr.update(), gr.update(visible=True, value=f"GroundingDINO requires text prompt.")
    image_np = np.array(input_image)
    boxes_filt, install_success = dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold)
    boxes_filt = boxes_filt.numpy()
    boxes_choice = [str(i) for i in range(boxes_filt.shape[0])]
    return Image.fromarray(show_boxes(image_np, boxes_filt.astype(int), show_index=True)), gr.update(choices=boxes_choice, value=boxes_choice), gr.update(visible=False) if install_success else gr.update(visible=True, value=f"GroundingDINO installment failed. Your process automatically fall back to local groundingdino. See your terminal for more detail and {dino_install_issue_text}")


def dino_batch_process(
    batch_sam_model_name, batch_dino_model_name, batch_text_prompt, batch_box_threshold, batch_dilation_amt,
    dino_batch_source_dir, dino_batch_dest_dir,
    dino_batch_output_per_image, dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask):
    if batch_text_prompt is None or batch_text_prompt == "":
        return "Please add text prompts to generate masks"
    print("Start batch processing")
    sam = init_sam_model(batch_sam_model_name)
    predictor = SamPredictorHQ(sam, 'hq' in batch_sam_model_name)

    if not os.path.exists(dino_batch_dest_dir):
        os.makedirs(dino_batch_dest_dir)
        print(f"Destination directory created: {dino_batch_dest_dir}")
    
    process_info = ""
    install_success = True
    all_files = glob.glob(os.path.join(dino_batch_source_dir, "*"))
    for image_index, input_image_file in enumerate(all_files):
        print(f"Processing {image_index}/{len(all_files)} {input_image_file}")
        try:
            input_image = Image.open(input_image_file).convert("RGBA")
        except:
            print(f"File {input_image_file} not image, skipped.")
            continue
        image_np = np.array(input_image)
        image_np_rgb = image_np[..., :3]

        boxes_filt, install_success = dino_predict_internal(input_image, batch_dino_model_name, batch_text_prompt, batch_box_threshold)
        if boxes_filt is None or boxes_filt.shape[0] == 0:
            msg = f"GroundingDINO generated 0 box for image {input_image_file}, please lower the box threshold if you want any segmentation for this image. "
            print(msg)
            process_info += (msg + "\n")
            continue
        
        predictor.set_image(image_np_rgb)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(sam_device),
            multimask_output=(dino_batch_output_per_image == 1))
        
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
        boxes_filt = boxes_filt.cpu().numpy().astype(int)
        
        create_mask_batch_output(
            input_image_file, dino_batch_dest_dir, 
            image_np, masks, boxes_filt, batch_dilation_amt, 
            dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask)
    
    garbage_collect(sam)
    return process_info + "Done" + ("" if install_success else f". However, GroundingDINO installment has failed. Your process automatically fall back to local groundingdino. See your terminal for more detail and {dino_install_issue_text}")


def cnet_seg(
    sam_model_name, cnet_seg_input_image, cnet_seg_processor, cnet_seg_processor_res, 
    cnet_seg_pixel_perfect, cnet_seg_resize_mode, target_W, target_H,
    auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area):
    print(f"Start semantic segmentation with processor {cnet_seg_processor}")
    auto_sam_output_mode = "coco_rle" if "seg" in cnet_seg_processor else "binary_mask"
    sam = load_sam_model(sam_model_name)
    predictor = SamPredictorHQ(sam, 'hq' in sam_model_name)
    register_auto_sam(predictor, auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area, auto_sam_output_mode)
    outputs = semantic_segmentation(cnet_seg_input_image, cnet_seg_processor, cnet_seg_processor_res,
                                    cnet_seg_pixel_perfect, cnet_seg_resize_mode, target_W, target_H)
    sem_sam_garbage_collect()
    garbage_collect(sam)
    return outputs


def image_layout(
    sam_model_name, layout_input_image_or_path, layout_output_path, 
    auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area):
    print("Start processing image layout")
    sam = load_sam_model(sam_model_name)
    predictor = SamPredictorHQ(sam, 'hq' in sam_model_name)
    register_auto_sam(predictor, auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area, "binary_mask")
    outputs = image_layer_internal(layout_input_image_or_path, layout_output_path)
    sem_sam_garbage_collect()
    garbage_collect(sam)
    return outputs


def categorical_mask(
    sam_model_name, crop_processor, crop_processor_res, 
    crop_pixel_perfect, crop_resize_mode, target_W, target_H,
    crop_category_input, crop_input_image, 
    auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area):
    print("Start processing categorical mask")
    sam = load_sam_model(sam_model_name)
    predictor = SamPredictorHQ(sam, 'hq' in sam_model_name)
    register_auto_sam(predictor, auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area, "coco_rle")
    outputs, resized_input_image = categorical_mask_image(crop_processor, crop_processor_res, crop_category_input, crop_input_image,
                                                          crop_pixel_perfect, crop_resize_mode, target_W, target_H)
    resized_input_image_pil = Image.fromarray(resized_input_image).convert("RGBA")
    resized_input_image_np = np.array(resized_input_image_pil)
    sem_sam_garbage_collect()
    garbage_collect(sam)
    if isinstance(outputs, str):
        return [], outputs, None
    output_gallery = create_mask_output(resized_input_image_np, outputs[None, None, ...], None)
    return output_gallery, "Done", resized_input_image_pil


def categorical_mask_batch(
    sam_model_name, crop_processor, crop_processor_res, 
    crop_pixel_perfect, crop_resize_mode, target_W, targe_H,
    crop_category_input, crop_batch_dilation_amt, crop_batch_source_dir, crop_batch_dest_dir, 
    crop_batch_save_image, crop_batch_save_mask, crop_batch_save_image_with_mask, crop_batch_save_background, 
    auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area):
    print("Start processing categorical mask in batch")
    sam = load_sam_model(sam_model_name)
    predictor = SamPredictorHQ(sam, 'hq' in sam_model_name)
    register_auto_sam(predictor, auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area, "coco_rle")
    all_files = glob.glob(os.path.join(crop_batch_source_dir, "*"))
    process_info = ""
    for image_index, input_image_file in enumerate(all_files):
        print(f"Processing {image_index}/{len(all_files)} {input_image_file}")
        try:
            crop_input_image = Image.open(input_image_file).convert("RGB")
        except:
            print(f"File {input_image_file} not image, skipped.")
            continue
        outputs, resized_input_image = categorical_mask_image(crop_processor, crop_processor_res, crop_category_input, crop_input_image, 
                                                              crop_pixel_perfect, crop_resize_mode, target_W, targe_H)
        if isinstance(outputs, str):
            outputs = f"Image {image_index}: {outputs}"
            print(outputs)
            process_info += outputs + "\n"
            continue
        resized_input_image_pil = Image.fromarray(resized_input_image).convert("RGBA")
        resized_input_image_np = np.array(resized_input_image_pil)
        create_mask_batch_output(
            input_image_file, crop_batch_dest_dir, 
            resized_input_image_np, outputs[None, None, ...], None, crop_batch_dilation_amt, 
            crop_batch_save_image, crop_batch_save_mask, crop_batch_save_background, crop_batch_save_image_with_mask)
    sem_sam_garbage_collect()
    garbage_collect(sam)
    return process_info + "Done"


def priorize_sam_scripts(is_img2img):
    cnet_idx, sam_idx = None, None
    if is_img2img:
        for idx, s in enumerate(scripts.scripts_img2img.alwayson_scripts):
            if s.title() == "Segment Anything":
                sam_idx = idx
            elif s.title() == "ControlNet":
                cnet_idx = idx
        if cnet_idx is not None and sam_idx is not None and cnet_idx < sam_idx:
            scripts.scripts_img2img.alwayson_scripts[cnet_idx], scripts.scripts_img2img.alwayson_scripts[
                sam_idx] = scripts.scripts_img2img.alwayson_scripts[sam_idx], scripts.scripts_img2img.alwayson_scripts[cnet_idx]
    else:
        for idx, s in enumerate(scripts.scripts_txt2img.alwayson_scripts):
            if s.title() == "Segment Anything":
                sam_idx = idx
            elif s.title() == "ControlNet":
                cnet_idx = idx
        if cnet_idx is not None and sam_idx is not None and cnet_idx < sam_idx:
            scripts.scripts_txt2img.alwayson_scripts[cnet_idx], scripts.scripts_txt2img.alwayson_scripts[
                sam_idx] = scripts.scripts_txt2img.alwayson_scripts[sam_idx], scripts.scripts_txt2img.alwayson_scripts[cnet_idx]


def ui_sketch_inner():
    sam_inpaint_color_sketch = gr.Image(label="Color sketch inpainting", source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA")
    sam_inpaint_mask_alpha = gr.Slider(label="Mask transparency")
    return sam_inpaint_color_sketch, sam_inpaint_mask_alpha    


def ui_sketch(sam_input_image, is_img2img):
    with gr.Column(visible=False):
        sam_sketch_checkbox = gr.Checkbox(value=False, label="Enable Sketch")
        with gr.Column(visible=False) as sketch_column:
            sam_inpaint_copy_button = gr.Button(value="Copy from input image")
            sam_inpaint_color_sketch, sam_inpaint_mask_alpha = ui_sketch_inner()
        sam_inpaint_copy_button.click(
                fn=lambda x: x,
                inputs=[sam_input_image],
                outputs=[sam_inpaint_color_sketch])
        sam_sketch_checkbox.change(
            fn=gr_show,
            inputs=[sam_sketch_checkbox],
            outputs=[sketch_column],
            show_progress=False)
    return sam_sketch_checkbox, sam_inpaint_color_sketch, sam_inpaint_mask_alpha

def ui_dilation(sam_output_mask_gallery, sam_output_chosen_mask, sam_input_image):
    sam_dilation_checkbox = gr.Checkbox(value=False, label="Expand Mask")
    with gr.Column(visible=False) as dilation_column:
        sam_dilation_amt = gr.Slider(minimum=0, maximum=100, value=0, label="Specify the amount that you wish to expand the mask by (recommend 30)")
        sam_dilation_output_gallery = gr.Gallery(label="Expanded Mask", columns=3)
        sam_dilation_submit = gr.Button(value="Update Mask")
        sam_dilation_submit.click(
            fn=update_mask,
            inputs=[sam_output_mask_gallery, sam_output_chosen_mask, sam_dilation_amt, sam_input_image],
            outputs=[sam_dilation_output_gallery])
    sam_dilation_checkbox.change(
        fn=gr_show,
        inputs=[sam_dilation_checkbox],
        outputs=[dilation_column],
        show_progress=False)
    return sam_dilation_checkbox, sam_dilation_output_gallery


def ui_inpaint(is_img2img, max_cn):
    with FormRow():
        if is_img2img:
            inpaint_upload_enable_label = "Copy to Inpaint Upload" + (" & img2img ControlNet Inpainting" if max_cn > 0 else "")
        else:
            inpaint_upload_enable_label = "Copy to txt2img ControlNet Inpainting" if max_cn > 0 else ""
        inpaint_upload_enable = gr.Checkbox(value=False, label=inpaint_upload_enable_label, visible=(len(inpaint_upload_enable_label) > 0))
        cnet_inpaint_invert = gr.Checkbox(value=False, label='ControlNet inpaint not masked', visible=((max_cn > 0) and not is_img2img))
        cnet_inpaint_idx = gr.Radio(value="0" if max_cn > 0 else None, choices=[str(i) for i in range(max_cn)], label='ControlNet Inpaint Index', type="index", visible=((max_cn > 0) and not is_img2img))
    return inpaint_upload_enable, cnet_inpaint_invert, cnet_inpaint_idx


def ui_batch(is_dino):
    dino_batch_dilation_amt = gr.Slider(minimum=0, maximum=100, value=0, label="Specify the amount that you wish to expand the mask by (recommend 0-10)")
    dino_batch_source_dir = gr.Textbox(label="Source directory")
    dino_batch_dest_dir = gr.Textbox(label="Destination directory")
    with gr.Row():
        dino_batch_output_per_image = gr.Radio(choices=["1", "3"], value="3", type="index", label="Output per image: ", visible=is_dino)
        dino_batch_save_image = gr.Checkbox(value=True, label="Save masked image")
        dino_batch_save_mask = gr.Checkbox(value=True, label="Save mask")
        dino_batch_save_image_with_mask = gr.Checkbox(value=True, label="Save original image with mask and bounding box")
        dino_batch_save_background = gr.Checkbox(value=False, label="Save background instead of foreground")
    dino_batch_run_button = gr.Button(value="Start batch process")
    dino_batch_progress = gr.Text(value="", label="GroundingDINO batch progress status")
    return dino_batch_dilation_amt, dino_batch_source_dir, dino_batch_dest_dir, dino_batch_output_per_image, dino_batch_save_image, dino_batch_save_mask, dino_batch_save_image_with_mask, dino_batch_save_background, dino_batch_run_button, dino_batch_progress


def ui_processor(use_random=True, use_cnet=True):
    processor_choices = ["seg_ufade20k", "seg_ofade20k", "seg_ofcoco"]
    if use_random:
        processor_choices.append("random")
    with gr.Row():
        cnet_seg_processor = gr.Radio(choices=processor_choices, value="seg_ufade20k", label="Choose preprocessor for semantic segmentation: ")
        cnet_seg_processor_res = gr.Slider(label="Preprocessor resolution", value=512, minimum=64, maximum=2048, step=1)
        cnet_seg_resize_mode = gr.Radio(choices=["Just Resize", "Crop and Resize", "Resize and Fill"], value="Crop and Resize", label="Resize Mode", type="index", visible=False)
        if use_random and use_cnet:
            cnet_seg_gallery_input = gr.Radio(
                choices=["1", "2"], value="2", type="index", visible=False, 
                label="Select ControlNet input from random segmentation gallery. Choose 2 for Edit-Anything ControlNet.")
        else:
            cnet_seg_gallery_input = gr.Label(visible=False)
    with gr.Row():
        cnet_seg_pixel_perfect = gr.Checkbox(value=False, label="Enable Pixel Perfect from lllyasviel. "
                                             "Configure your target width and height on txt2img/img2img default panel before preview if you wish to enable pixel perfect.")
        if use_random and use_cnet:
            cnet_seg_processor.change(
                fn=lambda x, y: (gr_show(x=="random"), gr_show(x!="random"), gr_show(x!="random" and not y), gr_show(x!="random" and y)),
                inputs=[cnet_seg_processor, cnet_seg_pixel_perfect],
                outputs=[cnet_seg_gallery_input, cnet_seg_pixel_perfect, cnet_seg_processor_res, cnet_seg_resize_mode],
                show_progress=False)
    cnet_seg_pixel_perfect.change(
        fn=lambda x: (gr_show(x), gr_show(not x)),
        inputs=[cnet_seg_pixel_perfect],
        outputs=[cnet_seg_resize_mode, cnet_seg_processor_res],
        show_progress=False)
    return cnet_seg_processor, cnet_seg_processor_res, cnet_seg_gallery_input, cnet_seg_pixel_perfect, cnet_seg_resize_mode
