# Some Utils for ComfyUI

## LoadImageWithSwitch
Modified the official LoadImage node by adding a switch. When turned off, it will not load the image.

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
