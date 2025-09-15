from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random



def generate_sample_digits(output_dir='data/synthetic/', n_samples=1000, font_size=28, font_dir='~/System/Library/Fonts/', include_blank_cells=True):
    import os
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    font_names = [
        'Helvetica',
        'Arial',
        'Verdana',
        'Times',
        'Courier'
    ]

    base_font_size = int(font_size * 1.25)
    fonts = [ImageFont.truetype(font_path, base_font_size) for font_path in font_names]

    samples_per_combination = n_samples // (9 * len(fonts))

    for digit in range(1, 10):
        for font in fonts:
            for sample in range(samples_per_combination):
                img_size = 44  # Slightly larger to avoid cropping
                img = Image.new('RGB', (img_size, img_size), 'white')
                draw = ImageDraw.Draw(img)

                size_variation = random.uniform(1.08, 1.15)
                varied_font = ImageFont.truetype(font.path, int(base_font_size * size_variation))

                # Keep numbers very close to center: minimal offset
                x_offset = random.uniform(-0.1, 0.1)
                y_offset = random.uniform(-0.1, 0.1)

                # Get text bounding box to center properly
                temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
                bbox = temp_draw.textbbox((0, 0), str(digit), font=varied_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Center the text
                x = (img_size - text_width) // 2 + x_offset
                y = (img_size - text_height) // 2 + y_offset

                # Less rotation
                rotation_angle = random.uniform(-3, 3)

                # Draw bold digit by overdrawing (dilution effect)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        draw.text((x + dx, y + dy), str(digit), font=varied_font, fill=(0, 0, 0, 180))

                overlay = Image.new('RGBA', (img_size, img_size), (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.text((x, y), str(digit), font=varied_font, fill=(255, 255, 255, 60))
                img = img.convert('RGBA')
                img = Image.alpha_composite(img, overlay).convert('RGB')

                if rotation_angle != 0:
                    img = img.rotate(rotation_angle, fillcolor='white')

                # Crop to 28x28 from center
                left = (img_size - 28) // 2
                top = (img_size - 28) // 2
                right = left + 28
                bottom = top + 28
                img = img.crop((left, top, right, bottom))

                img_array = np.array(img)

                dilution_strength = random.uniform(0.10, 0.22)
                img_array = (img_array * (1 - dilution_strength) + 255 * dilution_strength).astype(np.uint8)

                noise = np.random.normal(0, 6, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

                brightness = random.uniform(0.97, 1.04)
                contrast = random.uniform(0.97, 1.04)
                img_array = np.clip(img_array * brightness * contrast, 0, 255).astype(np.uint8)

                img = Image.fromarray(img_array).convert('L')

                filename = f"{output_dir}/digit_{digit}_font_{font_names[fonts.index(font)]}_sample_{sample:04d}.png"
                img.save(filename)

    if include_blank_cells:
        blank_samples = n_samples // 10
        for sample in range(blank_samples):
            img_size = 44
            img = Image.new('RGB', (img_size, img_size), 'white')
            img_array = np.array(img)

            noise = np.random.normal(0, 60, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

            for _ in range(random.randint(20, 40)):
                x = random.randint(0, img_size-1)
                y = random.randint(0, img_size-1)
                intensity = random.randint(30, 220)
                img_array[y, x] = [intensity, intensity, intensity]

            for _ in range(random.randint(2, 7)):
                area_size = random.randint(4, 14)
                start_x = random.randint(0, img_size - area_size)
                start_y = random.randint(0, img_size - area_size)
                for dy in range(area_size):
                    for dx in range(area_size):
                        if random.random() < 0.8:
                            x = start_x + dx
                            y = start_y + dy
                            if 0 <= x < img_size and 0 <= y < img_size:
                                intensity = random.randint(10, 180)
                                img_array[y, x] = [intensity, intensity, intensity]

            brightness = random.uniform(0.4, 1.6)
            contrast = random.uniform(0.5, 1.5)
            img_array = np.clip(img_array * brightness * contrast, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_array).convert('L')

            left = (img_size - 28) // 2
            top = (img_size - 28) // 2
            right = left + 28
            bottom = top + 28
            img = img.crop((left, top, right, bottom))

            filename = f"{output_dir}/digit_0_sample_{sample:04d}.png"
            img.save(filename)