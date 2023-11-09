import cv2
import diffusers
import os
import glob
from PIL import Image
from scripts.sdwebui import ControlNetUnit, i2i_inpaint_call, t2i_call
from scripts.easyphoto_config import (DEFAULT_NEGATIVE, DEFAULT_NEGATIVE_XL,
                                      DEFAULT_POSITIVE, DEFAULT_POSITIVE_XL,
                                      SDXL_MODEL_NAME,
                                      easyphoto_img2img_samples,
                                      easyphoto_outpath_samples,
                                      easyphoto_txt2img_samples, models_path,
                                      user_id_outpath_samples,
                                      validation_prompt)
import numpy as np

# Add control_mode=1 means Prompt is more important, to better control lips and eyes,
# this comments will be delete after 10 PR and for those who are not familiar with SDWebUIControlNetAPI
def get_controlnet_unit(unit, input_image, weight, use_preprocess=True):
    if unit == "canny":
        if use_preprocess:
            control_unit = ControlNetUnit(
                image=input_image,
                module="canny",
                weight=weight,
                guidance_end=1,
                control_mode=1,
                resize_mode="Just Resize",
                threshold_a=100,
                threshold_b=200,
                model="control_v11p_sd15_canny",
            )
            print("Processor is used for canny!")
            print(input_image.shape)
        else:
            # direct use the inout canny image with inner line
            control_unit = ControlNetUnit(
                image=input_image,
                module=None,
                weight=weight,
                guidance_end=1,
                control_mode=1,
                resize_mode="Crop and Resize",
                threshold_a=100,
                threshold_b=200,
                model="control_v11p_sd15_canny",
            )
            print("No processor is used for canny!")
            print(input_image.shape)
    elif unit == "openpose":
        control_unit = ControlNetUnit(
            image=input_image,
            module="openpose_full",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="control_v11p_sd15_openpose",
        )
    elif unit == "color":
        blur_ratio = 24
        h, w, c = np.shape(input_image)
        color_image = np.array(input_image, np.uint8)

        color_image = resize_image(color_image, 1024)
        now_h, now_w = color_image.shape[:2]

        color_image = cv2.resize(
            color_image,
            (int(now_w // blur_ratio), int(now_h // blur_ratio)),
            interpolation=cv2.INTER_CUBIC,
        )
        color_image = cv2.resize(
            color_image, (now_w, now_h), interpolation=cv2.INTER_NEAREST
        )
        color_image = cv2.resize(
            color_image, (w, h), interpolation=cv2.INTER_CUBIC)
        color_image = Image.fromarray(np.uint8(color_image))

        control_unit = ControlNetUnit(
            image=color_image,
            module="none",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="control_sd15_random_color",
        )
    elif unit == "tile":
        control_unit = ControlNetUnit(
            image=input_image,
            module="tile_resample",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            threshold_a=1,
            threshold_b=200,
            model="control_v11f1e_sd15_tile",
        )

    elif unit == "depth":
        control_unit = ControlNetUnit(
            image=input_image,
            module="depth_midas",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="control_v11f1p_sd15_depth",
        )
    elif unit=='ipa':
        control_unit = ControlNetUnit(
            image=input_image,
            module="ip-adapter_clip_sd15",
            weight=weight,
            guidance_end=1,
            control_mode=0,
            resize_mode="Just Resize",
            model="ip-adapter_sd15",
        )
    return control_unit


def inpaint(
    input_image: Image.Image,
    select_mask_input: Image.Image,
    controlnet_pairs: list,
    input_prompt="1girl",
    diffusion_steps=50,
    denoising_strength=0.45,
    hr_scale: float = 1.0,
    default_positive_prompt=DEFAULT_POSITIVE,
    default_negative_prompt=DEFAULT_NEGATIVE,
    seed: int = 123456,
    sd_model_checkpoint="Chilloutmix-Ni-pruned-fp16-fix.safetensors",
    sampler="DPM++ 2M SDE Karras",
):
    assert input_image is not None, f"input_image must not be none"
    controlnet_units_list = []
    w = int(input_image.width)
    h = int(input_image.height)

    # if controlnet_pairs:
    for pair in controlnet_pairs:
        controlnet_units_list.append(
            get_controlnet_unit(pair[0], pair[1], pair[2], pair[3])
        )

    positive = f"{input_prompt}, {default_positive_prompt}"
    negative = f"{default_negative_prompt}"

    image = i2i_inpaint_call(
        images=[input_image],
        mask_image=select_mask_input,
        inpainting_fill=1,
        steps=diffusion_steps,
        denoising_strength=denoising_strength,
        cfg_scale=7,
        inpainting_mask_invert=0,
        width=int(w * hr_scale),
        height=int(h * hr_scale),
        inpaint_full_res=False,
        seed=seed,
        prompt=positive,
        negative_prompt=negative,
        controlnet_units=controlnet_units_list,
        sd_model_checkpoint=sd_model_checkpoint,
        outpath_samples=easyphoto_img2img_samples,
        sampler=sampler,
    )

    return image


def txt2img(
    controlnet_pairs: list,
    input_prompt = '1girl',
    diffusion_steps = 50,
    width: int = 1024,
    height: int = 1024,
    default_positive_prompt = DEFAULT_POSITIVE,
    default_negative_prompt = DEFAULT_NEGATIVE,
    seed: int = 123456,
    sd_model_checkpoint = "Chilloutmix-Ni-pruned-fp16-fix.safetensors",
    sampler = "DPM++ 2M SDE Karras"
):
    controlnet_units_list = []

    for pair in controlnet_pairs:
        controlnet_units_list.append(
            get_controlnet_unit(pair[0], pair[1], pair[2])
        )

    positive = f'{input_prompt}, {default_positive_prompt}'
    negative = f'{default_negative_prompt}'

    image = t2i_call(
        steps=diffusion_steps,
        cfg_scale=7,
        width=width,
        height=height,
        seed=seed,
        prompt=positive,
        negative_prompt=negative,
        controlnet_units=controlnet_units_list,
        sd_model_checkpoint=sd_model_checkpoint,
        outpath_samples=easyphoto_txt2img_samples,
        sampler=sampler,
    )

    return image



def easyphoto_ipadapter_forward(sd_model_checkpoint, main_image, init_image, input_prompt):
    print(main_image)
    print(input_prompt)
    print(sd_model_checkpoint)

    seed = np.random.randint(0, 65536)

    model_path = os.path.join(
        models_path, f"Stable-diffusion", sd_model_checkpoint)
    # lora_path = os.path.join(models_path, f"Lora/{user_ids[0]}.safetensors")
    sd_base15_checkpoint = os.path.join(
        os.path.abspath(os.path.dirname(__file__)
                        ).replace("scripts", "models"),
        "stable-diffusion-v1-5",
    )

    main_image = Image.open(main_image).convert("RGB")
    template_image = Image.fromarray(np.uint8(init_image["image"]))  # template
    template_mask = Image.fromarray(np.uint8(init_image["mask"]))

    controlnet_pairs =  [
            ["ipa", np.array(main_image), 1.0, True],
        ]

    first_diffusion_steps=50
    first_denoising_strength=0.7

    result_img = inpaint(
        template_image,
        template_mask,
        controlnet_pairs,
        diffusion_steps=first_diffusion_steps,
        denoising_strength=first_denoising_strength,
        input_prompt=input_prompt,
        hr_scale=1.0,
        seed=str(seed),
        sd_model_checkpoint=sd_model_checkpoint,
    )


    result_img = np.uint8(result_img)
    cv2.imwrite('result_img.jpg',result_img)
    
    # return_res.append(main_image)

    return 'Success', [Image.fromarray(result_img)]