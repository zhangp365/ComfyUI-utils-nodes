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
Modified the official ImageCompositeMasked node by adding a switch. When turned off, it will not return the destination image directly.

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

## SplitMask
This node splits one mask into two masks of the same size according to the area of the submasks. If there are more than two areas, it will select the two largest submasks.

## MaskFastGrow
This node is designed for growing masks quickly. When using the official or other mask growth nodes, the speed slows down significantly with large grow values, such as above 20. In contrast, this node maintains consistent speed regardless of the grow value.

## MaskFromFaceModel
Generates a mask from the face model of the Reactor face swap node output. The mask covers the facial area below the eyes, excluding the forehead. If the forehead is crucial for your application, it's recommended to use a different mask or adjust the generated mask accordingly.

## MaskAutoSelector
Check the three input masks. If any are available, return the first. If none are available, raise an exception.

## MaskCoverFourCorners
Generates a mask by covering the selected corners with circular edges. This mask can be used as an attention mask to remove watermarks from the corners.

## CheckpointLoaderSimpleWithSwitch
Enhanced the official LoadCheckpoint node by integrating three switches. Each switch controls whether a specific component is loaded. When a switch is turned off, the corresponding component will not be loaded. if you use the extra vae and close the model's vae loading, that will save memory.

## ImageResizeTo8x
Modified the [image-resize-comfyui](https://github.com/palant/image-resize-comfyui) image resize node by adding logic to crop the resulting image size to 8 times size, similar to the VAE encode node. This avoids pixel differences when pasting back by the ImageCompositeMasked node.

## TextPreview
Added the node for convenience. The code is originally from ComfyUI-Custom-Scripts, thanks.

## MatchImageRatioToPreset
According to the input image ratio, decide which standard SDXL training size is the closest match. This is useful for subsequent image resizing and other processes.

## UpscaleImageWithModelIfNeed
Enhanced the official UpscaleImageWithModel node by adding a judge. If the input image area exceeds a predefined threshold, upscaling is bypassed. The threshold is a percentage of the SDXL standard size (1024x1024) area.