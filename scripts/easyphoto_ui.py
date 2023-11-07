import glob
import os
import time

import gradio as gr
import requests
import numpy as np
from PIL import Image
from modules import script_callbacks, shared
import modules
from scripts.easyphoto_config import (cache_log_file_path, models_path,
                                      user_id_outpath_samples,easyphoto_outpath_samples)
from scripts.easyphoto_infer import easyphoto_infer_forward
from scripts.easyphoto_train import easyphoto_train_forward
from scripts.easyphoto_utils import check_id_valid
from modules import ui_common
from modules.ui_components import ToolButton as ToolButton_webui
import modules.generation_parameters_copypaste as parameters_copypaste

gradio_compat = True

try:
    from distutils.version import LooseVersion

    from importlib_metadata import version
    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass

# def create_output_panel(tabname, outdir):
#     return ui_common.create_output_panel(tabname, outdir)

def get_external_ckpts():
    external_checkpoints = []
    external_ckpt_dir = shared.cmd_opts.ckpt_dir if shared.cmd_opts.ckpt_dir else []
    if len(external_ckpt_dir) > 0:
        for _checkpoint in os.listdir(external_ckpt_dir):
            if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                external_checkpoints.append(_checkpoint)
    return external_checkpoints
external_checkpoints = get_external_ckpts()

def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths

def refresh_display():
    if not os.path.exists(os.path.dirname(cache_log_file_path)):
        os.makedirs(os.path.dirname(cache_log_file_path), exist_ok=True)
    lines_limit = 3
    try:
        with open(cache_log_file_path, "r", newline="") as f:
            lines = []
            for s in f.readlines():
                line = s.replace("\x00", "")
                if line.strip() == "" or line.strip() == "\r":
                    continue
                lines.append(line)

            total_lines = len(lines)
            if total_lines <= lines_limit:
                chatbot = [(None, ''.join(lines))]
            else:
                chatbot = [(None, ''.join(lines[total_lines-lines_limit:]))]
            return chatbot
    except Exception:
        with open(cache_log_file_path, "w") as f:
            pass
        return None

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", 
                         elem_classes=kwargs.pop('elem_classes', []) + ["cnet-toolbutton"], 
                         **kwargs)

    def get_block_name(self):
        return "button"



def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as easyphoto_tabs:
        with gr.TabItem('Train'):
            dummy_component = gr.Label(visible=False)
            with gr.Blocks():
                with gr.Row():
                    uuid = gr.Text(label="User_ID", value="", visible=False)

                    with gr.Column():
                        gr.Markdown('Main Image')
                        main_image = gr.Image(label="Main Image", elem_id="{id_part}_image", show_label=False, source="upload", type="filepath")

                    with gr.Column():
                        gr.Markdown('Training photos')

                        instance_images = gr.Gallery().style(columns=[4], rows=[2], object_fit="contain", height="auto")

                        with gr.Row():
                            upload_button = gr.UploadButton(
                                "Upload Photos", file_types=["image"], file_count="multiple"
                            )
                            clear_button = gr.Button("Clear Photos")
                        clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)

                        upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images, queue=False)
                    
                        gr.Markdown(
                            '''
                            Training steps:
                            1. Please upload a main image (clean image with only the main target), and 5-20 training images (image contain the main target).
                            2. Click on the Start Training button below to start the training process, approximately 10 minutes on A10*1.
                            3. Switch to Inference and generate photos based on the template. 
                            4. If you encounter lag when uploading, please modify the size of the uploaded pictures and try to limit it to 1.5MB.
                            '''
                        )
                    with gr.Column():
                        gr.Markdown('Params Setting')
                        with gr.Accordion("Advanced Options", open=True):
                            with gr.Row():
                                def checkpoint_refresh_function():
                                    checkpoints = []
                                    for _checkpoint in os.listdir(os.path.join(models_path, "Stable-diffusion")):
                                        if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                                            checkpoints.append(_checkpoint)
                                    return gr.update(choices=list(set(["Chilloutmix-Ni-pruned-fp16-fix.safetensors"] + checkpoints + external_checkpoints)))
                                
                                checkpoints = []
                                for _checkpoint in os.listdir(os.path.join(models_path, "Stable-diffusion")):
                                    if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                                        checkpoints.append(_checkpoint)
                                sd_model_checkpoint = gr.Dropdown(value="Chilloutmix-Ni-pruned-fp16-fix.safetensors", choices=list(set(["Chilloutmix-Ni-pruned-fp16-fix.safetensors"] + checkpoints + external_checkpoints)), label="The base checkpoint you use.", visible=True)

                                checkpoint_refresh = ToolButton(value="\U0001f504")
                                checkpoint_refresh.click(
                                    fn=checkpoint_refresh_function,
                                    inputs=[],
                                    outputs=[sd_model_checkpoint]
                                )

                            with gr.Row():
                                resolution = gr.Textbox(
                                    label="resolution",
                                    value=512,
                                    interactive=True
                                )
                                val_and_checkpointing_steps = gr.Textbox(
                                    label="validation & save steps",
                                    value=100,
                                    interactive=True
                                )
                                max_train_steps = gr.Textbox(
                                    label="max train steps",
                                    value=800,
                                    interactive=True
                                )
                                steps_per_photos = gr.Textbox(
                                    label="max steps per photos",
                                    value=200,
                                    interactive=True
                                )

                            with gr.Row():
                                train_batch_size = gr.Textbox(
                                    label="train batch size",
                                    value=1,
                                    interactive=True
                                )
                                gradient_accumulation_steps = gr.Textbox(
                                    label="gradient accumulationsteps",
                                    value=4,
                                    interactive=True
                                )
                                dataloader_num_workers =  gr.Textbox(
                                    label="dataloader num workers",
                                    value=16,
                                    interactive=True
                                )
                                learning_rate = gr.Textbox(
                                    label="learning rate",
                                    value=1e-4,
                                    interactive=True
                                )
                            with gr.Row():
                                rank = gr.Textbox(
                                    label="rank",
                                    value=128,
                                    interactive=True
                                )
                                network_alpha = gr.Textbox(
                                    label="network alpha",
                                    value=64,
                                    interactive=True
                                )
                            with gr.Row():
                                refine_mask = gr.Checkbox(
                                    label="Refine Mask",  
                                    value=True
                                )
                                use_mask = gr.Checkbox(
                                    label="Use Mask", 
                                    value=True
                                )

                            with gr.Row():
                                validation = gr.Checkbox(
                                    label="Validation",  
                                    value=False,
                                    visible=False
                                )
                                enable_rl = gr.Checkbox(
                                    label="Enable RL (Reinforcement Learning)",
                                    value=False,
                                    visible=False
                                )
                            
                            # Reinforcement Learning Options
                            with gr.Row(visible=False) as rl_option_row1:
                                max_rl_time = gr.Slider(
                                    minimum=1, maximum=12, value=1,
                                    step=0.5, label="max time (hours) of RL"
                                )
                                timestep_fraction = gr.Slider(
                                    minimum=0.7, maximum=1, value=1,
                                    step=0.05, label="timestep fraction"
                                )
                            rl_notes = gr.Markdown(
                                value = '''
                                RL notes:
                                - The RL is an experimental feature aiming to improve the face similarity score of generated photos w.r.t uploaded photos.
                                - Setting (**max rl time** / **timestep fraction**) > 2 is recommended for a stable training result.
                                - 16GB GPU memory is required at least.
                                ''',
                                visible=False
                            )
                            enable_rl.change(lambda x: rl_option_row1.update(visible=x), inputs=[enable_rl], outputs=[rl_option_row1])
                            enable_rl.change(lambda x: rl_option_row1.update(visible=x), inputs=[enable_rl], outputs=[rl_notes])

                        gr.Markdown(
                            '''
                            Parameter parsing:
                            - **max steps per photo** represents the maximum number of training steps per photo.
                            - **max train steps** represents the maximum training step.
                            - **Validation** Whether to validate at training time.
                            - Final training step = Min(photo_num * max_steps_per_photos, max_train_steps)
                            '''
                        )

                with gr.Row():
                    with gr.Column(width=3):
                        run_button = gr.Button('Start Training')
                    with gr.Column(width=1):
                        refresh_button = gr.Button('Refresh Log')

                gr.Markdown(
                    '''
                    We need to train first to predict, please wait for the training to complete, thank you for your patience.  
                    '''
                )
                output_message  = gr.Markdown()

                with gr.Box():
                    logs_out        = gr.Chatbot(label='Training Logs', height=200)
                    block           = gr.Blocks()
                    with block:
                        block.load(refresh_display, None, logs_out, every=3)

                    refresh_button.click(
                        fn = refresh_display,
                        inputs = [],
                        outputs = [logs_out]
                    )

                run_button.click(fn=easyphoto_train_forward,
                                _js="ask_for_style_name",
                                inputs=[
                                    sd_model_checkpoint, dummy_component,
                                    uuid,
                                    resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, \
                                    learning_rate, rank, network_alpha, validation, main_image, instance_images, \
                                    enable_rl, max_rl_time, timestep_fraction,refine_mask,use_mask
                                ],
                                outputs=[output_message])
                                
        with gr.TabItem('Inference'):
            dummy_component = gr.Label(visible=False)
            training_templates = glob.glob(os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), 'training_templates/*.jpg'))
            infer_templates = glob.glob(os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), 'infer_templates/*.jpg'))
            preset_template = list(training_templates) + list(infer_templates)

            with gr.Blocks() as demo:
                with gr.Row():
                    with gr.Column():
                        model_selected_tab = gr.State(0)
                        init_image = gr.Image(label="Image for skybox", elem_id="{id_part}_image", show_label=False, source="upload", tool='sketch')
                        # init_image = gr.Image(label="Image for skybox", elem_id="{id_part}_image", show_label=False, source="upload")
                        
                        with gr.Row():
                            def checkpoint_refresh_function():
                                checkpoints = []
                                for _checkpoint in os.listdir(os.path.join(models_path, "Stable-diffusion")):
                                    if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                                        checkpoints.append(_checkpoint)
                                return gr.update(choices=list(set(["Chilloutmix-Ni-pruned-fp16-fix.safetensors"] + checkpoints + external_checkpoints)))
                            
                            checkpoints = []
                            for _checkpoint in os.listdir(os.path.join(models_path, "Stable-diffusion")):
                                if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                                    checkpoints.append(_checkpoint)
                            sd_model_checkpoint = gr.Dropdown(value="Chilloutmix-Ni-pruned-fp16-fix.safetensors", choices=list(set(["Chilloutmix-Ni-pruned-fp16-fix.safetensors"] + checkpoints + external_checkpoints)), label="The base checkpoint you use.", visible=True)

                            checkpoint_refresh = ToolButton(value="\U0001f504")
                            checkpoint_refresh.click(
                                fn=checkpoint_refresh_function,
                                inputs=[],
                                outputs=[sd_model_checkpoint]
                            )

                        with gr.Row():
                            def select_function():
                                ids = []
                                if os.path.exists(user_id_outpath_samples):
                                    _ids = os.listdir(user_id_outpath_samples)
                                    for _id in _ids:
                                        if check_id_valid(_id, user_id_outpath_samples, models_path):
                                            ids.append(_id)
                                ids = sorted(ids)
                                return gr.update(choices=["none"] + ids)

                            ids = []
                            if os.path.exists(user_id_outpath_samples):
                                _ids = os.listdir(user_id_outpath_samples)
                                for _id in _ids:
                                    if check_id_valid(_id, user_id_outpath_samples, models_path):
                                        ids.append(_id)
                                ids = sorted(ids)

                            num_of_faceid = gr.Dropdown(value=str(1), elem_id='dropdown', choices=[1, 2, 3, 4, 5], label=f"Num of Faceid", visible=False)

                            uuids           = []
                            visibles        = [True, False, False, False, False]
                            for i in range(int(5)):
                                uuid = gr.Dropdown(value="none", elem_id='dropdown', choices=["none"] + ids, min_width=140, label=f"User_{i} id", visible=visibles[i])
                                uuids.append(uuid)

                            def update_uuids(_num_of_faceid):
                                _uuids = []
                                for i in range(int(_num_of_faceid)):
                                    _uuids.append(gr.update(value="none", visible=True))
                                for i in range(int(5 - int(_num_of_faceid))):
                                    _uuids.append(gr.update(value="none", visible=False))
                                return _uuids
                            
                            num_of_faceid.change(update_uuids, inputs=[num_of_faceid], outputs=uuids)
                            
                            refresh = ToolButton(value="\U0001f504")
                            for i in range(int(5)):
                                refresh.click(
                                    fn=select_function,
                                    inputs=[],
                                    outputs=[uuids[i]]
                                )

                        with gr.Row():
                            match_and_paste = gr.Checkbox(
                                label="Match and Paste",  
                                value=True
                            )
                            
                            remove_target = gr.Checkbox(
                                label="Remove Target",  
                                value=False
                            )

                        with gr.Accordion("Advanced Options", open=False):
                            additional_prompt = gr.Textbox(
                                label="Additional Prompt",
                                lines=3,
                                value='masterpiece, beauty',
                                interactive=True
                            )
                            seed = gr.Textbox(
                                label="Seed", 
                                value=-1,
                            )
                            with gr.Row():
                                first_diffusion_steps = gr.Slider(
                                    minimum=15, maximum=50, value=50,
                                    step=1, label='Diffusion steps'
                                )
                                first_denoising_strength = gr.Slider(
                                    minimum=0.40, maximum=1.00, value=0.70,
                                    step=0.05, label='Diffusion denoising strength'
                                )

                            with gr.Row():
                                lora_weight = gr.Slider(
                                    minimum=0, maximum=1, value=0.8,
                                    step=0.1, label='LoRA weight'
                                )
                                iou_threshold = gr.Slider(
                                    minimum=0, maximum=1, value=0.7,
                                    step=0.05, label='IoU Threshold '
                                )

                            with gr.Row():
                                angle = gr.Slider(
                                    minimum=-90, maximum=90, value=0.0,
                                    step=1, label='Angle'
                                )
                                azimuth  = gr.Slider(
                                    minimum=-60, maximum=60, value=0.0,
                                    step=1, label='Azimuth'
                                )
                                ratio = gr.Slider(
                                    minimum=0.5, maximum=5.5, value=1.0,
                                    step=0.1, label='Ratio'
                                )

                            with gr.Row():
                                batch_size = gr.Slider(
                                    minimum=1, maximum=10, value=1,
                                    step=1, label='Batch Size'
                                )
                               

                            with gr.Row():
                                refine_input_mask = gr.Checkbox(
                                    label="Refine Input Mask",  
                                    value=True
                                )
                                optimize_angle_and_ratio = gr.Checkbox(
                                    label="Optimize Angle and Ratio", 
                                    value=True
                                )
                                refine_bound = gr.Checkbox(
                                    label="Refine Boundary",  
                                    value=True
                                )
                                pure_image = gr.Checkbox(
                                    label="Pure Image",  
                                    value=True
                                )
                                global_inpaint = gr.Checkbox(
                                    label="Global Inpaint", 
                                    value=False
                                )

                                # change_shape = gr.Checkbox(
                                #     label="Change Shape",  
                                #     value=True
                                # )
                                # optimize_vertex = gr.Checkbox(
                                #     label="Optimize Vertex",  
                                #     value=True
                                # )
                                # use_dragdiffusion = gr.Checkbox(
                                #     label="Use Dragdiffusion",  
                                #     value=True
                                # )
                            
                        display_button = gr.Button('Start Generation')

                    with gr.Column():
                        gr.Markdown('Generated Results')
      
                        output_images = gr.Gallery(
                            label='Output',
                            show_label=False
                        ).style(columns=[4], rows=[2], object_fit="contain", height="auto")

                        with gr.Row():
                            tabname = 'easyphoto'
                            buttons = {
                                'img2img': ToolButton_webui('🖼️', elem_id=f'{tabname}_send_to_img2img', tooltip="Send image and generation parameters to img2img tab."),
                                'inpaint': ToolButton_webui('🎨️', elem_id=f'{tabname}_send_to_inpaint', tooltip="Send image and generation parameters to img2img inpaint tab."),
                                'extras': ToolButton_webui('📐', elem_id=f'{tabname}_send_to_extras', tooltip="Send image and generation parameters to extras tab.")
                            }

                        for paste_tabname, paste_button in buttons.items():
                            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                                paste_button=paste_button, tabname=paste_tabname, source_tabname="txt2img" if tabname == "txt2img" else None, source_image_component=output_images,
                                paste_field_names=[]
                            ))


                        infer_progress = gr.Textbox(
                            label="Generation Progress",
                            value="No task currently",
                            interactive=False
                        )


            
                display_button.click(
                    fn=easyphoto_infer_forward,
                    inputs=[sd_model_checkpoint, init_image, additional_prompt, seed, first_diffusion_steps, first_denoising_strength, \
                            lora_weight, iou_threshold, angle, azimuth, ratio, batch_size, refine_input_mask, optimize_angle_and_ratio, refine_bound, \
                            pure_image, global_inpaint, match_and_paste, remove_target, model_selected_tab, *uuids],
                            
                    outputs=[infer_progress, output_images]
                )
            
    return [(easyphoto_tabs, "EasyPhoto", f"EasyPhoto_tabs")]

# 注册设置页的配置项
def on_ui_settings():
    section = ('EasyPhoto', "EasyPhoto")
    shared.opts.add_option("easyphoto_cache_model", shared.OptionInfo(
        True, "Cache preprocess model in Inference", gr.Checkbox, {}, section=section))

script_callbacks.on_ui_settings(on_ui_settings)  # 注册进设置页
script_callbacks.on_ui_tabs(on_ui_tabs)
