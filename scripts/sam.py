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
from modules import scripts, shared, script_callbacks
from modules.ui import gr_show
from modules.ui_components import FormRow, ResizeHandleRow
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessing
from modules.devices import device
from scripts.dino import dino_model_list
from scripts.process_params import SAMProcessUnit, max_cn_num
from lib_segment_anything.sam import (priorize_sam_scripts, refresh_symbol, ToolButton, refresh_sam_models,
    dino_predict, sam_predict, ui_inpaint, ui_dilation, get_sam_model_list, dino_batch_process, ui_batch,
    cnet_seg, ui_processor, image_layout, categorical_mask, categorical_mask_batch, clear_cache
)
from lib_segment_anything.api import sam_api


class Script(scripts.Script):

    def title(self):
        return 'Segment Anything'

    def show(self, is_img2img):
        needHide = specialTabForUI() and not is_img2img
        return scripts.AlwaysVisible if not needHide else False

    def ui(self, is_img2img):
        if max_cn_num() > 0:
            priorize_sam_scripts(is_img2img)
        tab_name = "img2img" if is_img2img else "txt2img"
        tab_prefix = tab_name + "_sam_"
        ui_process = ()
        with (gr.Accordion('Segment Anything', open=False, elem_id=f"segment_anything_accordion_{tab_name}") if not specialTabForUI()
              else gr.Group(elem_id=f"segment_anything_accordion_{tab_name}")):
            with gr.Row():
                with gr.Column(scale=10):
                    with gr.Row():
                        sam_model_list = get_sam_model_list()
                        sam_model_name = gr.Dropdown(label="SAM Model", choices=sam_model_list, value=sam_model_list[0] if len(sam_model_list) > 0 else None)
                        sam_refresh_models = ToolButton(value=refresh_symbol)
                        sam_refresh_models.click(refresh_sam_models, sam_model_name, sam_model_name)
                with gr.Column(scale=1):
                    sam_use_cpu = gr.Checkbox(value=False, label="Use CPU for SAM")
                    def change_sam_device(use_cpu=False):
                        global sam_device
                        sam_device = "cpu" if use_cpu else device
                    sam_use_cpu.change(fn=change_sam_device, inputs=[sam_use_cpu], show_progress=False)
            with gr.Tabs():
                with gr.TabItem(label="Single Image"):
                    with (gr.Group() if not specialTabForUI() else ResizeHandleRow()):
                        with (gr.Group() if not specialTabForUI() else gr.Column()):
                            gr.HTML(value="<p>Left click the image to add one positive point (black dot). Right click the image to add one negative point (red dot). Left click the point to remove it.</p>")
                            sam_input_image = gr.Image(label="Image for Segment Anything", elem_id=f"{tab_prefix}input_image", source="upload", type="pil", image_mode="RGBA")
                            sam_remove_dots = gr.Button(value="Remove all point prompts")
                            sam_dummy_component = gr.Label(visible=False)
                            sam_remove_dots.click(
                                fn=lambda _: None,
                                _js="samRemoveDots",
                                inputs=[sam_dummy_component],
                                outputs=None)
                            gr.HTML(value="<p>GroundingDINO + Segment Anything can achieve [text prompt]->[object detection]->[segmentation]</p>")
                            dino_checkbox = gr.Checkbox(value=False, label="Enable GroundingDINO", elem_id=f"{tab_prefix}dino_enable_checkbox")
                            with gr.Column(visible=False) as dino_column:
                                gr.HTML(value="<p>Due to the limitation of Segment Anything, when there are point prompts, at most 1 box prompt will be allowed; when there are multiple box prompts, no point prompts are allowed.</p>")
                                dino_model_name = gr.Dropdown(label="GroundingDINO Model (Auto download from huggingface)", choices=dino_model_list, value=dino_model_list[0])
                                dino_text_prompt = gr.Textbox(placeholder="You must enter text prompts to enable groundingdino. Otherwise this extension will fall back to point prompts only.", label="GroundingDINO Detection Prompt", elem_id=f"{tab_prefix}dino_text_prompt")
                                dino_box_threshold = gr.Slider(label="GroundingDINO Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001)
                                dino_preview_checkbox = gr.Checkbox(value=False, label="I want to preview GroundingDINO detection result and select the boxes I want.", elem_id=f"{tab_prefix}dino_preview_checkbox")
                                with gr.Column(visible=False) as dino_preview:
                                    dino_preview_boxes = gr.Image(show_label=False, type="pil", image_mode="RGBA")
                                    dino_preview_boxes_button = gr.Button(value="Generate bounding box", elem_id=f"{tab_prefix}dino_run_button")
                                    dino_preview_boxes_selection = gr.CheckboxGroup(label="Select your favorite boxes: ", elem_id=f"{tab_prefix}dino_preview_boxes_selection")
                                    dino_preview_result = gr.Text(value="", label="GroundingDINO preview status", visible=False)
                                    dino_preview_boxes_button.click(
                                        fn=dino_predict,
                                        _js="submit_dino",
                                        inputs=[sam_input_image, dino_model_name, dino_text_prompt, dino_box_threshold],
                                        outputs=[dino_preview_boxes, dino_preview_boxes_selection, dino_preview_result])
                                dino_preview_checkbox.change(
                                    fn=gr_show,
                                    inputs=[dino_preview_checkbox],
                                    outputs=[dino_preview],
                                    show_progress=False)
                            dino_checkbox.change(
                                fn=gr_show,
                                inputs=[dino_checkbox],
                                outputs=[dino_column],
                                show_progress=False)

                        with (gr.Group() if not specialTabForUI() else gr.Column()):
                            sam_output_mask_gallery = gr.Gallery(label='Segment Anything Output', columns=3)
                            sam_submit = gr.Button(value="Preview Segmentation", elem_id=f"{tab_prefix}run_button")
                            sam_result = gr.Text(value="", label="Segment Anything status")
                            sam_submit.click(
                                fn=sam_predict,
                                _js='submit_sam',
                                inputs=[sam_model_name, sam_input_image,        # SAM
                                        sam_dummy_component, sam_dummy_component,   # Point prompts
                                        dino_checkbox, dino_model_name, dino_text_prompt, dino_box_threshold,  # DINO prompts
                                        dino_preview_checkbox, dino_preview_boxes_selection],  # DINO preview prompts
                                outputs=[sam_output_mask_gallery, sam_result])
                            with FormRow():
                                sam_output_chosen_mask = gr.Radio(label="Choose your favorite mask: ", value="0", choices=["0", "1", "2"], type="index")
                                gr.Checkbox(value=False, label="Preview automatically when add/remove points", elem_id=f"{tab_prefix}realtime_preview_checkbox")
                            sam_inpaint_upload_enable, sam_cnet_inpaint_invert, sam_cnet_inpaint_idx = ui_inpaint(is_img2img, max_cn_num())
                            sam_dilation_checkbox, sam_dilation_output_gallery = ui_dilation(sam_output_mask_gallery, sam_output_chosen_mask, sam_input_image)
                            sam_single_image_process = (
                                sam_inpaint_upload_enable, sam_cnet_inpaint_invert, sam_cnet_inpaint_idx,
                                sam_input_image, sam_output_mask_gallery, sam_output_chosen_mask, 
                                sam_dilation_checkbox, sam_dilation_output_gallery)
                            ui_process += sam_single_image_process

                with gr.TabItem(label="Batch Process"):
                    with (gr.Group() if not specialTabForUI() else ResizeHandleRow()):
                        with (gr.Group() if not specialTabForUI() else gr.Column()):
                            gr.Markdown(value="You may configurate the following items and generate masked image for all images under a directory. This mode is designed for generating LoRA/LyCORIS training set.")
                            gr.Markdown(value="The current workflow is [text prompt]->[object detection]->[segmentation]. Semantic segmentation support is in Auto SAM panel.")
                            dino_batch_model_name = gr.Dropdown(label="GroundingDINO Model (Auto download from huggingface)", choices=dino_model_list, value=dino_model_list[0])
                            dino_batch_text_prompt = gr.Textbox(label="GroundingDINO Detection Prompt")
                            dino_batch_box_threshold = gr.Slider(label="GroundingDINO Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001)
                        with (gr.Group() if not specialTabForUI() else gr.Column()):
                            dino_batch_dilation_amt, dino_batch_source_dir, dino_batch_dest_dir, dino_batch_output_per_image, dino_batch_save_image, dino_batch_save_mask, dino_batch_save_image_with_mask, dino_batch_save_background, dino_batch_run_button, dino_batch_progress = ui_batch(True)
                    dino_batch_run_button.click(
                        fn=dino_batch_process,
                        inputs=[sam_model_name, dino_batch_model_name, dino_batch_text_prompt, dino_batch_box_threshold, dino_batch_dilation_amt,
                                dino_batch_source_dir, dino_batch_dest_dir, dino_batch_output_per_image, 
                                dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask],
                        outputs=[dino_batch_progress])
                    
                with gr.TabItem(label="Auto SAM"):
                    gr.Markdown("Auto SAM is mainly for semantic segmentation and image layout generation, which is supported based on ControlNet. You must have ControlNet extension installed, and you should not change its directory name (sd-webui-controlnet).")
                    gr.Markdown("The annotator directory inside the SAM extension directory is only a symbolic link. This is to save your space and make the extension repository clean.")

                    with gr.Accordion(label="Auto SAM Config", open=False):
                        gr.Markdown("You may configurate automatic sam generation. See [here](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L35-L96) for explanation of each parameter. If you still cannot understand, use default.")
                        with gr.Row():
                            auto_sam_points_per_side = gr.Number(label="points_per_side", value=32, precision=0)
                            auto_sam_points_per_batch = gr.Number(label="points_per_batch", value=64, precision=0)
                            auto_sam_pred_iou_thresh = gr.Slider(minimum=0, maximum=1, value=0.88, step=0.01, label="pred_iou_thresh")
                            auto_sam_stability_score_thresh = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.01, label="stability_score_thresh")
                            auto_sam_stability_score_offset = gr.Number(label="stability_score_offset", value=1)
                        with gr.Row():
                            auto_sam_box_nms_thresh = gr.Slider(label="box_nms_thresh", value=0.7, minimum=0, maximum=1, step=0.01)
                            auto_sam_crop_n_layers = gr.Number(label="crop_n_layers", value=0, precision=0)
                            auto_sam_crop_nms_thresh = gr.Slider(label="crop_nms_thresh", value=0.7, minimum=0, maximum=1, step=0.01)
                            auto_sam_crop_overlap_ratio = gr.Slider(label="crop_overlap_ratio", value=512/1500, minimum=0, maximum=1, step=0.01)
                            auto_sam_crop_n_points_downscale_factor = gr.Number(label="crop_n_points_downscale_factor", value=1, precision=0)
                        auto_sam_min_mask_region_area = gr.Number(label="min_mask_region_area", value=0, precision=0)
                        auto_sam_config = (auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
                                        auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
                                        auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
                                        auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area)

                    with gr.Tabs():
                        with gr.TabItem(label="ControlNet"):
                            with (gr.Group() if not specialTabForUI() else ResizeHandleRow()):
                                with (gr.Group() if not specialTabForUI() else gr.Column()):
                                    gr.Markdown(
                                        "You can enhance semantic segmentation for control_v11p_sd15_seg from lllyasviel. "
                                        "You can also utilize [Edit-Anything](https://github.com/sail-sg/EditAnything) and generate images according to random segmentation which preserve image layout.")
                                    cnet_seg_processor, cnet_seg_processor_res, cnet_seg_gallery_input, cnet_seg_pixel_perfect, cnet_seg_resize_mode = ui_processor(use_cnet=(max_cn_num() > 0))
                                    cnet_seg_input_image = gr.Image(label="Image for Auto Segmentation", source="upload", type="pil", image_mode="RGBA")
                                with (gr.Group() if not specialTabForUI() else gr.Column()):
                                    cnet_seg_output_gallery = gr.Gallery(label="Auto segmentation output", columns=2)
                                    cnet_seg_submit = gr.Button(value="Preview segmentation image")
                                    cnet_seg_status = gr.Text(value="", label="Segmentation status")
                                    cnet_seg_submit.click(
                                        fn=cnet_seg,
                                        inputs=[sam_model_name, cnet_seg_input_image, cnet_seg_processor, cnet_seg_processor_res, cnet_seg_pixel_perfect, cnet_seg_resize_mode, img2img_width if is_img2img else txt2img_width, img2img_height if is_img2img else txt2img_height, *auto_sam_config],
                                        outputs=[cnet_seg_output_gallery, cnet_seg_status])
                                    with gr.Row(visible=(max_cn_num() > 0)):
                                        cnet_seg_enable_copy = gr.Checkbox(value=False, label='Copy to ControlNet Segmentation')
                                        cnet_seg_idx = gr.Radio(value="0" if max_cn_num() > 0 else None, choices=[str(i) for i in range(max_cn_num())], label='ControlNet Segmentation Index', type="index")
                                    auto_sam_process = (cnet_seg_output_gallery, cnet_seg_enable_copy, cnet_seg_idx, cnet_seg_gallery_input)
                                    ui_process += auto_sam_process

                        with gr.TabItem(label="Image Layout"):
                            gr.Markdown("You can generate image layout either in single image or in batch. Since there might be A LOT of outputs, there is no gallery for preview. You need to go to the output folder for either single image or batch process.")
                            layout_mode = gr.Radio(choices=["single image", "batch process"], value="single image", type="index", label="Choose mode: ")
                            layout_input_image = gr.Image(label="Image for Image Layout", source="upload", type="pil", image_mode="RGBA")
                            layout_input_path = gr.Textbox(label="Input path", placeholder="Enter input path", visible=False)
                            layout_output_path = gr.Textbox(label="Output path", placeholder="Enter output path")
                            layout_submit_single = gr.Button(value="Generate layout for single image")
                            layout_submit_batch = gr.Button(value="Generate layout for batch process", visible=False)
                            layout_status = gr.Text(value="", label="Image layout status")
                            def layout_show(mode):
                                is_single = mode == 0
                                return gr_show(is_single), gr_show(is_single), gr_show(not is_single), gr_show(not is_single)
                            layout_mode.change(
                                fn=layout_show,
                                inputs=[layout_mode],
                                outputs=[layout_input_image, layout_submit_single, layout_input_path, layout_submit_batch])
                            layout_submit_single.click(
                                fn=image_layout,
                                inputs=[sam_model_name, layout_input_image, layout_output_path, *auto_sam_config],
                                outputs=[layout_status])
                            layout_submit_batch.click(
                                fn=image_layout,
                                inputs=[sam_model_name, layout_input_path, layout_output_path, *auto_sam_config],
                                outputs=[layout_status])

                        with gr.TabItem(label="Mask by Category"):
                            gr.Markdown(
                                "You can mask images by their categories via semantic segmentation. Please enter category ids (integers), separated by `+`. "
                                "Visit [here](https://github.com/Mikubill/sd-webui-controlnet/blob/main/annotator/oneformer/oneformer/data/datasets/register_ade20k_panoptic.py#L12-L207) for ade20k "
                                "and [here](https://github.com/Mikubill/sd-webui-controlnet/blob/main/annotator/oneformer/detectron2/data/datasets/builtin_meta.py#L20-L153) for coco to get category->id map. Note that coco jumps some numbers, so the actual ID is line_number - 21.")
                            crop_processor, crop_processor_res, _, crop_pixel_perfect, crop_resize_mode = ui_processor(False)
                            crop_category_input = gr.Textbox(placeholder="Enter categody ids, separated by +. For example, if you want bed+person, your input should be 7+12 for ade20k and 59+0 for coco.", label="Enter category IDs")
                            with gr.Tabs():
                                with gr.TabItem(label="Single Image"):
                                    with gr.Row():
                                        with gr.Column():
                                            crop_input_image = gr.Image(label="Image to be masked", source="upload", type="pil", image_mode="RGBA")
                                        with gr.Column():
                                            crop_output_gallery = gr.Gallery(label="Output", columns=3)
                                            crop_padding = gr.Number(value=-2, visible=False, interactive=False, precision=0)
                                            crop_resized_image = gr.Image(label="Resized image", source="upload", type="pil", image_mode="RGBA", visible=False)
                                            crop_submit = gr.Button(value="Preview mask")
                                            crop_result = gr.Text(value="", label="Categorical mask status")
                                    crop_submit.click(
                                        fn=categorical_mask,
                                        inputs=[sam_model_name, crop_processor, crop_processor_res, crop_pixel_perfect, crop_resize_mode, 
                                                img2img_width if is_img2img else txt2img_width, img2img_height if is_img2img else txt2img_height, 
                                                crop_category_input, crop_input_image, *auto_sam_config],
                                        outputs=[crop_output_gallery, crop_result, crop_resized_image])
                                    crop_inpaint_enable, crop_cnet_inpaint_invert, crop_cnet_inpaint_idx = ui_inpaint(is_img2img, max_cn_num())
                                    crop_dilation_checkbox, crop_dilation_output_gallery = ui_dilation(crop_output_gallery, crop_padding, crop_resized_image)
                                    crop_single_image_process = (
                                        crop_inpaint_enable, crop_cnet_inpaint_invert, crop_cnet_inpaint_idx, 
                                        crop_resized_image, crop_output_gallery, crop_padding, 
                                        crop_dilation_checkbox, crop_dilation_output_gallery)
                                    ui_process += crop_single_image_process

                                with gr.TabItem(label="Batch Process"):
                                    crop_batch_dilation_amt, crop_batch_source_dir, crop_batch_dest_dir, _, crop_batch_save_image, crop_batch_save_mask, crop_batch_save_image_with_mask, crop_batch_save_background, crop_batch_run_button, crop_batch_progress = ui_batch(False)
                                    crop_batch_run_button.click(
                                        fn=categorical_mask_batch,
                                        inputs=[sam_model_name, crop_processor, crop_processor_res, crop_pixel_perfect, crop_resize_mode, 
                                                img2img_width if is_img2img else txt2img_width, img2img_height if is_img2img else txt2img_height, 
                                                crop_category_input, crop_batch_dilation_amt, crop_batch_source_dir, crop_batch_dest_dir, 
                                                crop_batch_save_image, crop_batch_save_mask, crop_batch_save_image_with_mask, crop_batch_save_background, *auto_sam_config],
                                        outputs=[crop_batch_progress])
                            
                            
                with gr.TabItem(label="Upload Mask to ControlNet Inpainting"):
                    gr.Markdown("This panel is for those who want to upload mask to ControlNet inpainting. It is not part of the SAM feature. It might be removed someday when ControlNet support uploading image and mask. "
                                "It serves as a temporarily workaround to overcome the unavailability of image with mask uploading feature in ControlNet extension.")
                    with gr.Row():
                        cnet_upload_enable = gr.Checkbox(value=False, label="Enable uploading manually created mask to SAM.")
                        cnet_upload_num = gr.Radio(value="0", choices=[str(i) for i in range(max_cn_num())], label='ControlNet Inpaint Number', type="index")
                    with gr.Column(visible=False) as cnet_upload_panel:
                        cnet_upload_img_inpaint = gr.Image(label="Image for ControlNet Inpaint", show_label=False, source="upload", interactive=True, type="pil")
                        cnet_upload_mask_inpaint = gr.Image(label="Mask for ControlNet Inpaint", source="upload", interactive=True, type="pil")
                    cnet_upload_enable.change(
                        fn=gr_show,
                        inputs=[cnet_upload_enable],
                        outputs=[cnet_upload_panel],
                        show_progress=False)
                    cnet_upload_process = (cnet_upload_enable, cnet_upload_num, cnet_upload_img_inpaint, cnet_upload_mask_inpaint)
                    ui_process += cnet_upload_process

                with gr.Row():
                    switch = gr.Button(value="Switch to Inpaint Upload")
                    unload = gr.Button(value="Unload all models from memory")
                    uncheck = gr.Button(value="Uncheck all copies")
                    switch.click(
                        fn=lambda _: None,
                        _js="switchToInpaintUpload",
                        inputs=[sam_dummy_component],
                        outputs=None)
                    unload.click(
                        fn=clear_cache,
                        inputs=[],
                        outputs=[])
                    uncheck.click(
                        fn=lambda: (gr.update(value=False), gr.update(value=False), gr.update(value=False)),
                        inputs=[],
                        outputs=[sam_inpaint_upload_enable, cnet_seg_enable_copy, crop_inpaint_enable],
                        show_progress=False)
        
        return ui_process

    def process(self, p: StableDiffusionProcessing, *args):
        is_img2img = isinstance(p, StableDiffusionProcessingImg2Img)
        process_unit = SAMProcessUnit(args, is_img2img)
        process_unit.set_process_attributes(p)


def on_after_component(component, **_kwargs):
    global txt2img_width
    if getattr(component, 'elem_id', None) == 'txt2img_width':
        txt2img_width = component
        return

    global txt2img_height
    if getattr(component, 'elem_id', None) == 'txt2img_height':
        txt2img_height = component
        return

    global img2img_width
    if getattr(component, 'elem_id', None) == 'img2img_width':
        img2img_width = component
        return

    global img2img_height
    if getattr(component, 'elem_id', None) == 'img2img_height':
        img2img_height = component
        return



def on_ui_settings():
    section = ('segment_anything', "Segment Anything")
    shared.opts.add_option("sam_use_local_groundingdino", shared.OptionInfo(False, "Use local groundingdino to bypass C++ problem", section=section))
    shared.opts.add_option("sam_special_tab_for_ui", shared.OptionInfo(False, "Move UI into a special tab instead of txt2img/img2img", section=section))


def specialTabForUI():
    return shared.opts.data.get("sam_special_tab_for_ui", False)

def onUITabs():
    tab = gr.Blocks(elem_id="segment_anything_tab")
    return [(tab, "Segment Anything", "segment_anything")]


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(on_after_component)
if specialTabForUI():
    script_callbacks.on_ui_tabs(onUITabs)
script_callbacks.on_app_started(sam_api)
