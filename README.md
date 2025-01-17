# Some Utils for ComfyUI

## LoadImageWithSwitch
Modified the official LoadImage node by adding a switch. When turned off, it will not load the image.

## LoadImageMaskWithSwitch
Modified the official LoadImageMask node by adding a switch. When turned off, it will not load the image to mask.

## LoadImageWithoutListDir
When there are a lot of images in the input directory, loading image with `os.listdir` can be slow. This node avoids using `os.listdir` to improve performance.

## LoadImageMaskWithoutListDir
When there are a lot of images in the input directory, loading image as Mask with `os.listdir` can be slow. This node avoids using `os.listdir` to improve performance.

## ImageCompositeMaskedWithSwitch
Modified the official ImageCompositeMasked node by adding a switch. When turned off, it will return the destination image directly.

## ImageCompositeMaskedOneByOne
Modified the official ImageCompositeMasked node to process images one by one, instead of processing an entire batch at once. In video scenarios, processing in a batch may requires a significant amount of memory, but this method helps reduce memory usage.

## ImageBatchOneOrMore
This node can input one or more images, the limit is six. It expands the functionality of the official ImageBatch node from two to multiple images.

## ImageConcatenateOfUtils

This node, ImageConcatenateOfUtils, is an extension of the original [ImageConcatenate](https://github.com/kijai/ComfyUI-KJNodes) node developed by @kijai.

### Features
- **Upscale**: This extension adds the capability to upscale images.
- **Check**: Additional functionality for cheching the second image empty or not.

### Original node
The original ImageConcatenate node can be found [here](https://github.com/kijai/ComfyUI-KJNodes).
Special thanks to @kijai for their contribution to the initial version.

## ColorCorrectOfUtils
This node, ColorCorrectOfUtils, is an extension of the original [ColorCorrect](https://github.com/EllangoK/ComfyUI-post-processing-nodes/blob/master/post_processing/color_correct.py) node developed by @EllangoK. Added the chanels of red, green, and blue adjustment functionalities.

## ModifyTextGender
This node adjusts the text to describe the gender based on the input. If the gender input is 'M', the text will be adjusted to describe as male; if the gender input is 'F', it will be adjusted to describe as female.

## GenderControlOutput
This node determines the output based on the input gender. If the gender input is 'M', it will output male-specific text, float, and integer values. If the gender input is 'F', it will output female-specific text, float, and integer values.

## BooleanControlOutput
This node outputs different values based on a boolean input. If the boolean input is True, it will output the values of true_text, true_float, true_int, True, and False. If the boolean input is False, it will output the values of false_text, false_float, false_int, False, and True.

## SplitMask
This node splits one mask into two masks of the same size according to the area of the submasks. If there are more than two areas, it will select the two largest submasks.

## MaskFastGrow
This node is designed for growing masks quickly. When using the official or other mask growth nodes, the speed slows down significantly with large grow values, such as above 20. In contrast, this node maintains consistent speed regardless of the grow value.

## MaskFromFaceModel
Generates a mask from the face model of the Reactor face swap node. The mask covers the facial area below the eyes, excluding the forehead. Enabling add_bbox_upper_points provides a rough approximation but lacks precision. If the forehead is essential for your application, consider using a different mask or adjusting the generated mask as needed.

<img src="assets/maskFromFacemodel.png" width="100%"/>

## MaskAutoSelector
Check the three input masks. If any are available, return the first. If none are available, raise an exception.

## MaskCoverFourCorners
Generates a mask by covering the selected corners with circular edges. This mask can be used as an attention mask to remove watermarks from the corners.

## MaskofCenter
Generates a mask by covering the center of the image with a circular edge. This mask can be used as an attention mask, then model can focus on the center of the image.

## CheckpointLoaderSimpleWithSwitch
Enhanced the official LoadCheckpoint node by integrating three switches. Each switch controls whether a specific component is loaded. When a switch is turned off, the corresponding component will not be loaded. if you use the extra vae and close the model's vae loading, that will save memory.

## ImageResizeTo8x
Modified the [image-resize-comfyui](https://github.com/palant/image-resize-comfyui) image resize node by adding logic to crop the resulting image size to 8 times size, similar to the VAE encode node. This avoids pixel differences when pasting back by the ImageCompositeMasked node.

## ImageAutoSelector
This node is designed to automatically select the image from the input. If the prior image is not empty, return the prior image; otherwise, return the alternative image or the third image.

## TextPreview
Added the node for convenience. The code is originally from ComfyUI-Custom-Scripts, thanks.

## TextInputAutoSelector
Check the component and alternative input. If the component input is not empty, return this text; otherwise, return the alternative text.

## MatchImageRatioToPreset
According to the input image ratio, decide which standard SDXL training size is the closest match. This is useful for subsequent image resizing and other processes.

## UpscaleImageWithModelIfNeed
Enhanced the official UpscaleImageWithModel node by adding a judge. If the input image area exceeds a predefined threshold, upscaling is bypassed. The threshold is a percentage of the SDXL standard size (1024x1024) area.

## ImageCompositeWatermark
This node is designed to composite a watermark into the destination image. It can select the position of the watermark, resize the watermark according to the input ratio, and add a margin to the watermark.

## ImageTransition
This node is designed to generate a transition image between two images. It can generate a transition image between two images.

## TorchCompileModelAdvanced
This node enables model compilation using torch.compile. It extends ComfyUI's original torch compile node by adding compile mode options and a toggle switch.

## DetectorForNSFW
This node adapts the original model and inference code from  [nudenet](https://github.com/notAI-tech/NudeNet.git) for use with Comfy. A small 10MB default model, [320n.onnx](https://github.com/notAI-tech/NudeNet?tab=readme-ov-file#available-models), is provided. If you wish to use other models from that repository, download the  [ONNX model](https://github.com/notAI-tech/NudeNet?tab=readme-ov-file#available-models) and place it in the models/nsfw directory, then set the appropriate detect_size.

From initial testing, the filtering effect is better than classifier models such as [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection).

<img src="assets/detectorForNSFW.png" width="100%"/>
You can also adjust the confidence levels for various rules such as buttocks_exposed to be more lenient or strict. Lower confidence levels will filter out more potential NSFW images. Setting the value to 1 will stop filtering for that specific feature.

## DeepfaceAnalyzeFaceAttributes
This node integrates the [deepface](https://github.com/serengil/deepface) library to analyze face attributes (gender, race, emotion, age). It analyzes only the largest face in the image and supports processing one image at a time.
<img src="assets/deepfaceAnalyzeFaceAttributes.png" width="100%"/>

If the input image is a standard square face image, you can enable the standard_single_face_image switch. In this case, the node will skip face detection and analyze the attributes directly.

Upon the first run, the node will download the [deepface](https://github.com/serengil/deepface) models, which may take some time.

> **Note:** If you encounter the following exception while running the node:

> ```
> ValueError: The layer sequential has never been called and thus has no defined input.
> ```

> Please set the environment variable `TF_USE_LEGACY_KERAS` to `1`, then restart ComfyUI.
