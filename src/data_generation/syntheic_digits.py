from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random



def generate_sample_digits(output_dir='data/synthetic/', n_samples=1000, font_size=20, font_dir='~/System/Library/Fonts/', include_blank_cells=True):
    import os
    from pathlib import Path

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    font_names = [
        'Helvetica',
        'Arial',
        'Verdana',
        'Times',
        'Courier'
    ]

    fonts = [ImageFont.truetype(font_path, font_size) for font_path in font_names]

    # Calculate samples per digit/font combination
    samples_per_combination = n_samples // (9 * len(fonts))

    for digit in range(1, 10):
        for font in fonts:
            for sample in range(samples_per_combination):
                # Create larger image to prevent cutting off during rotation
                img_size = 40  # Increased from 28 to 40
                img = Image.new('RGB', (img_size, img_size), 'white')
                draw = ImageDraw.Draw(img)

                # Wider font size variation for more diversity
                size_variation = random.uniform(0.6, 1.4)
                varied_font = ImageFont.truetype(font.path, int(font_size * size_variation))

                # Add random variations with decimal precision
                x_offset = random.uniform(-4.0, 4.0)
                y_offset = random.uniform(-4.0, 4.0)

                # Get text bounding box to center properly
                temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
                bbox = temp_draw.textbbox((0, 0), str(digit), font=varied_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Center the text properly
                x = (img_size - text_width) // 2 + x_offset
                y = (img_size - text_height) // 2 + y_offset

                # Wider rotation range for more variation
                rotation_angle = random.uniform(-15, 15)

                # Draw text with rotation
                if rotation_angle != 0:
                    # Create temporary image for rotation
                    temp_img = Image.new('RGB', (img_size, img_size), 'white')
                    temp_draw = ImageDraw.Draw(temp_img)
                    temp_draw.text((x, y), str(digit), font=varied_font, fill='black')
                    img = temp_img.rotate(rotation_angle, fillcolor='white')
                else:
                    draw.text((x, y), str(digit), font=varied_font, fill='black')

                # Crop back to 28x28 from center
                left = (img_size - 28) // 2
                top = (img_size - 28) // 2
                right = left + 28
                bottom = top + 28
                img = img.crop((left, top, right, bottom))

                # Add less noise and distortion for clearer digits
                img_array = np.array(img)

                # Reduce Gaussian noise stddev for clarity
                noise = np.random.normal(0, 3, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

                # Narrower brightness/contrast adjustment for less distortion
                brightness = random.uniform(0.9, 1.1)
                contrast = random.uniform(0.95, 1.1)
                img_array = np.clip(img_array * brightness * contrast, 0, 255).astype(np.uint8)

                # Convert back to PIL Image
                img = Image.fromarray(img_array).convert('L')

                # Save with unique filename
                filename = f"{output_dir}/digit_{digit}_font_{font_names[fonts.index(font)]}_sample_{sample:04d}.png"
                img.save(filename)

    # Generate blank cells if requested
    if include_blank_cells:
        blank_samples = n_samples // 10  # 10% of samples are blank cells
        for sample in range(blank_samples):
            # Create larger image to prevent cutting off during rotation
            img_size = 40
            img = Image.new('RGB', (img_size, img_size), 'white')

            # Add random noise patterns to simulate empty cells
            img_array = np.array(img)

            # Add various types of noise
            # Gaussian noise
            noise = np.random.normal(0, 20, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

            # Add some random spots/speckles
            for _ in range(random.randint(5, 15)):
                x = random.randint(0, img_size-1)
                y = random.randint(0, img_size-1)
                intensity = random.randint(50, 200)
                img_array[y, x] = [intensity, intensity, intensity]

            # Add larger black areas (shadows, stains, etc.)
            for _ in range(random.randint(0, 3)):  # 0-3 large areas per image
                # Random size and position for black area
                area_size = random.randint(3, 8)
                start_x = random.randint(0, img_size - area_size)
                start_y = random.randint(0, img_size - area_size)

                # Create irregular shaped area
                for dy in range(area_size):
                    for dx in range(area_size):
                        if random.random() < 0.7:  # 70% chance to add pixel to area
                            x = start_x + dx
                            y = start_y + dy
                            if 0 <= x < img_size and 0 <= y < img_size:
                                # Vary intensity for more realistic look
                                intensity = random.randint(20, 120)
                                img_array[y, x] = [intensity, intensity, intensity]

            # Random brightness/contrast adjustment
            brightness = random.uniform(0.6, 1.4)
            contrast = random.uniform(0.7, 1.3)
            img_array = np.clip(img_array * brightness * contrast, 0, 255).astype(np.uint8)

            # Convert back to PIL Image
            img = Image.fromarray(img_array).convert('L')

            # Crop back to 28x28 from center
            left = (img_size - 28) // 2
            top = (img_size - 28) // 2
            right = left + 28
            bottom = top + 28
            img = img.crop((left, top, right, bottom))

            # Save blank cell
            filename = f"{output_dir}/digit_0_sample_{sample:04d}.png"
            img.save(filename)