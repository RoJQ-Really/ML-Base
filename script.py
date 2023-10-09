# This Algorithm Draw gradient map with parameters
from PIL import Image, ImageDraw, ImageFont


def create_background(size: int, font_path: str, block_size: float):
    new_image = Image.new("RGB", (size, size))  # create square
    draw_image = ImageDraw.Draw(new_image)
    block_size_px = size * block_size  # size one block
    max_color_value = 255  # max color's in palette
    tumbler = True  # tumbler for work while method
    current_x = 0  # x - cords in cycle
    current_y = 0  # y - cords in cycle
    color_delta = (block_size_px / size) * max_color_value
    while tumbler:

        RedValue = int((current_x / block_size_px) * color_delta)
        BlueValue = int((current_y / block_size_px) * color_delta)
        GreenValue = int((((current_x > current_y) * current_x + (current_x <= current_y) * current_y) / block_size_px) * color_delta)
        ColorPack = (RedValue, GreenValue, BlueValue)
        draw_image.rectangle((current_x, current_y, current_x + block_size_px, current_y + block_size_px), fill=ColorPack)
        if current_x + block_size_px == size and current_y + block_size_px == size:
            tumbler = False

        elif current_x + block_size_px == size:
            current_y += block_size_px
            current_x = 0

        elif current_x + block_size_px < size:
            current_x += block_size_px
        print(current_x, current_y, ColorPack)
    new_image.show("Test")


if __name__ == '__main__':
    create_background(1000, "", 0.01)






