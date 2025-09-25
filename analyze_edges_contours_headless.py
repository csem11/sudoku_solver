#!/usr/bin/env python3
"""
Edge/Contour Analysis for Processed Digit Images (Headless version)
Analyzes processed cell images to count edges/contours per digit
"""

import cv2 as cv
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import csv

# Set up paths
processed_dir = Path("data/digits/manual/processed")
output_dir = Path("data/analysis")

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

print("=== Edge/Contour Analysis for Processed Digit Images ===")
print(f"Processing images from: {processed_dir}")
print(f"Output directory: {output_dir}")

# Check if processed directory exists
if not processed_dir.exists():
    print(f"Error: Processed directory {processed_dir} does not exist!")
    exit(1)

# Get all processed image files
image_files = list(processed_dir.glob("*.jpg"))
print(f"Found {len(image_files)} processed images")

if len(image_files) == 0:
    print("No processed images found!")
    exit(1)

def extract_digit_from_filename(filename):
    """Extract digit from filename like '5_g0_c23_man_processed.jpg'"""
    try:
        parts = filename.stem.split('_')
        if len(parts) >= 1:
            return int(parts[0])
    except (ValueError, IndexError):
        pass
    return None

def count_edges_and_contours(image):
    """
    Count edges and contours in a processed digit image.
    Returns a dictionary with various edge/contour metrics.
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Edge detection using Canny
    edges = cv.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    
    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours (likely noise)
    min_contour_area = 5
    significant_contours = [c for c in contours if cv.contourArea(c) > min_contour_area]
    
    # Calculate total contour perimeter
    total_perimeter = sum(cv.arcLength(c, True) for c in significant_contours)
    
    # Calculate total contour area
    total_contour_area = sum(cv.contourArea(c) for c in significant_contours)
    
    return {
        'edge_pixels': edge_pixels,
        'num_contours': len(significant_contours),
        'total_perimeter': total_perimeter,
        'total_contour_area': total_contour_area,
        'avg_contour_area': total_contour_area / max(len(significant_contours), 1)
    }

# Initialize data storage
digit_stats = defaultdict(list)
processed_count = 0
error_count = 0

print("\nProcessing images...")
for img_file in image_files:
    # Extract digit from filename
    digit = extract_digit_from_filename(img_file)
    if digit is None:
        print(f"Warning: Could not extract digit from {img_file.name}")
        error_count += 1
        continue
    
    # Load and process image
    try:
        image = cv.imread(str(img_file), cv.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not load image {img_file.name}")
            error_count += 1
            continue
        
        # Count edges and contours
        stats = count_edges_and_contours(image)
        stats['filename'] = img_file.name
        
        # Store stats by digit
        digit_stats[digit].append(stats)
        processed_count += 1
        
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} images...")
            
    except Exception as e:
        print(f"Error processing {img_file.name}: {e}")
        error_count += 1

print(f"\nProcessing complete!")
print(f"Successfully processed: {processed_count} images")
print(f"Errors encountered: {error_count} images")
print(f"Digits found: {sorted(digit_stats.keys())}")

# Calculate averages per digit
digit_averages = {}

print("\n=== AVERAGE EDGE/CONTOUR COUNTS PER DIGIT ===")
print("Digit | Count | Avg Edges | Avg Contours | Avg Perimeter | Avg Area")
print("-" * 70)

for digit in sorted(digit_stats.keys()):
    stats_list = digit_stats[digit]
    count = len(stats_list)
    
    if count > 0:
        avg_edges = np.mean([s['edge_pixels'] for s in stats_list])
        avg_contours = np.mean([s['num_contours'] for s in stats_list])
        avg_perimeter = np.mean([s['total_perimeter'] for s in stats_list])
        avg_area = np.mean([s['total_contour_area'] for s in stats_list])
        
        digit_averages[digit] = {
            'count': count,
            'avg_edges': avg_edges,
            'avg_contours': avg_contours,
            'avg_perimeter': avg_perimeter,
            'avg_area': avg_area,
            'std_edges': np.std([s['edge_pixels'] for s in stats_list]),
            'std_contours': np.std([s['num_contours'] for s in stats_list]),
            'std_perimeter': np.std([s['total_perimeter'] for s in stats_list]),
            'std_area': np.std([s['total_contour_area'] for s in stats_list])
        }
        
        print(f"  {digit}   | {count:5d} | {avg_edges:8.1f} | {avg_contours:10.1f} | {avg_perimeter:12.1f} | {avg_area:8.1f}")

print("\n" + "=" * 70)

# Show summary statistics
total_images = sum(len(stats_list) for stats_list in digit_stats.values())
print(f"\nSUMMARY:")
print(f"Total images analyzed: {total_images}")
print(f"Digits represented: {len(digit_stats)}")
print(f"Average images per digit: {total_images / len(digit_stats):.1f}")

# Find digit with most/least edges
if digit_averages:
    most_edges_digit = max(digit_averages.keys(), key=lambda d: digit_averages[d]['avg_edges'])
    least_edges_digit = min(digit_averages.keys(), key=lambda d: digit_averages[d]['avg_edges'])
    
    print(f"\nDigit with most edges: {most_edges_digit} ({digit_averages[most_edges_digit]['avg_edges']:.1f} avg)")
    print(f"Digit with least edges: {least_edges_digit} ({digit_averages[least_edges_digit]['avg_edges']:.1f} avg)")
    
    most_contours_digit = max(digit_averages.keys(), key=lambda d: digit_averages[d]['avg_contours'])
    least_contours_digit = min(digit_averages.keys(), key=lambda d: digit_averages[d]['avg_contours'])
    
    print(f"Digit with most contours: {most_contours_digit} ({digit_averages[most_contours_digit]['avg_contours']:.1f} avg)")
    print(f"Digit with least contours: {least_contours_digit} ({digit_averages[least_contours_digit]['avg_contours']:.1f} avg)")

# Create visualizations
if digit_averages:
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Edge/Contour Analysis by Digit', fontsize=16, fontweight='bold')
    
    digits = sorted(digit_averages.keys())
    
    # Plot 1: Average Edge Pixels
    avg_edges = [digit_averages[d]['avg_edges'] for d in digits]
    std_edges = [digit_averages[d]['std_edges'] for d in digits]
    
    axes[0, 0].bar(digits, avg_edges, yerr=std_edges, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[0, 0].set_title('Average Edge Pixels per Digit')
    axes[0, 0].set_xlabel('Digit')
    axes[0, 0].set_ylabel('Average Edge Pixels')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (digit, avg, std) in enumerate(zip(digits, avg_edges, std_edges)):
        axes[0, 0].text(digit, avg + std + 5, f'{avg:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Average Number of Contours
    avg_contours = [digit_averages[d]['avg_contours'] for d in digits]
    std_contours = [digit_averages[d]['std_contours'] for d in digits]
    
    axes[0, 1].bar(digits, avg_contours, yerr=std_contours, capsize=5, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    axes[0, 1].set_title('Average Number of Contours per Digit')
    axes[0, 1].set_xlabel('Digit')
    axes[0, 1].set_ylabel('Average Number of Contours')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (digit, avg, std) in enumerate(zip(digits, avg_contours, std_contours)):
        axes[0, 1].text(digit, avg + std + 0.1, f'{avg:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Average Perimeter
    avg_perimeter = [digit_averages[d]['avg_perimeter'] for d in digits]
    std_perimeter = [digit_averages[d]['std_perimeter'] for d in digits]
    
    axes[1, 0].bar(digits, avg_perimeter, yerr=std_perimeter, capsize=5, alpha=0.7, color='lightcoral', edgecolor='darkred')
    axes[1, 0].set_title('Average Contour Perimeter per Digit')
    axes[1, 0].set_xlabel('Digit')
    axes[1, 0].set_ylabel('Average Perimeter')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (digit, avg, std) in enumerate(zip(digits, avg_perimeter, std_perimeter)):
        axes[1, 0].text(digit, avg + std + 10, f'{avg:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Sample Count per Digit
    counts = [digit_averages[d]['count'] for d in digits]
    
    axes[1, 1].bar(digits, counts, alpha=0.7, color='gold', edgecolor='orange')
    axes[1, 1].set_title('Number of Samples per Digit')
    axes[1, 1].set_xlabel('Digit')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (digit, count) in enumerate(zip(digits, counts)):
        axes[1, 1].text(digit, count + 5, f'{count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / "edge_contour_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_path}")
    
    plt.close()  # Close the figure to free memory

# Save detailed results to CSV
csv_path = output_dir / "edge_contour_analysis.csv"
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['digit', 'count', 'avg_edges', 'std_edges', 'avg_contours', 'std_contours', 
                  'avg_perimeter', 'std_perimeter', 'avg_area', 'std_area']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for digit in sorted(digit_averages.keys()):
        row = {
            'digit': digit,
            'count': digit_averages[digit]['count'],
            'avg_edges': digit_averages[digit]['avg_edges'],
            'std_edges': digit_averages[digit]['std_edges'],
            'avg_contours': digit_averages[digit]['avg_contours'],
            'std_contours': digit_averages[digit]['std_contours'],
            'avg_perimeter': digit_averages[digit]['avg_perimeter'],
            'std_perimeter': digit_averages[digit]['std_perimeter'],
            'avg_area': digit_averages[digit]['avg_area'],
            'std_area': digit_averages[digit]['std_area']
        }
        writer.writerow(row)

print(f"Detailed results saved to: {csv_path}")

# Save summary report
report_path = output_dir / "edge_contour_report.txt"
with open(report_path, 'w') as f:
    f.write("EDGE/CONTOUR ANALYSIS REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Analysis Date: {np.datetime64('now')}\n")
    f.write(f"Total Images Processed: {processed_count}\n")
    f.write(f"Errors Encountered: {error_count}\n")
    f.write(f"Digits Analyzed: {len(digit_stats)}\n\n")
    
    f.write("AVERAGE METRICS PER DIGIT:\n")
    f.write("-" * 30 + "\n")
    f.write("Digit | Count | Avg Edges | Avg Contours | Avg Perimeter | Avg Area\n")
    f.write("-" * 70 + "\n")
    
    for digit in sorted(digit_averages.keys()):
        stats = digit_averages[digit]
        f.write(f"  {digit}   | {stats['count']:5d} | {stats['avg_edges']:8.1f} | "
                f"{stats['avg_contours']:10.1f} | {stats['avg_perimeter']:12.1f} | "
                f"{stats['avg_area']:8.1f}\n")
    
    f.write("\nKEY INSIGHTS:\n")
    f.write("-" * 15 + "\n")
    
    if digit_averages:
        most_edges = max(digit_averages.keys(), key=lambda d: digit_averages[d]['avg_edges'])
        least_edges = min(digit_averages.keys(), key=lambda d: digit_averages[d]['avg_edges'])
        most_contours = max(digit_averages.keys(), key=lambda d: digit_averages[d]['avg_contours'])
        least_contours = min(digit_averages.keys(), key=lambda d: digit_averages[d]['avg_contours'])
        
        f.write(f"• Digit with most edges: {most_edges} ({digit_averages[most_edges]['avg_edges']:.1f} avg)\n")
        f.write(f"• Digit with least edges: {least_edges} ({digit_averages[least_edges]['avg_edges']:.1f} avg)\n")
        f.write(f"• Digit with most contours: {most_contours} ({digit_averages[most_contours]['avg_contours']:.1f} avg)\n")
        f.write(f"• Digit with least contours: {least_contours} ({digit_averages[least_contours]['avg_contours']:.1f} avg)\n")
        
        # Calculate complexity ranking
        complexity_scores = {}
        for digit in digit_averages.keys():
            # Simple complexity score combining edges and contours
            edges_norm = digit_averages[digit]['avg_edges'] / max(digit_averages[d]['avg_edges'] for d in digit_averages.keys())
            contours_norm = digit_averages[digit]['avg_contours'] / max(digit_averages[d]['avg_contours'] for d in digit_averages.keys())
            complexity_scores[digit] = (edges_norm + contours_norm) / 2
        
        sorted_by_complexity = sorted(complexity_scores.items(), key=lambda x: x[1], reverse=True)
        
        f.write(f"\nCOMPLEXITY RANKING (based on edges + contours):\n")
        for i, (digit, score) in enumerate(sorted_by_complexity, 1):
            f.write(f"{i:2d}. Digit {digit}: {score:.3f}\n")
        
        # ZERO DETECTION OPTIMIZATION ANALYSIS
        f.write(f"\nZERO DETECTION OPTIMIZATION:\n")
        f.write("-" * 30 + "\n")
        
        zero_stats = digit_averages[0]
        f.write(f"Digit 0 characteristics:\n")
        f.write(f"• Average edges: {zero_stats['avg_edges']:.1f} ± {zero_stats['std_edges']:.1f}\n")
        f.write(f"• Average contours: {zero_stats['avg_contours']:.1f} ± {zero_stats['std_contours']:.1f}\n")
        f.write(f"• Average perimeter: {zero_stats['avg_perimeter']:.1f} ± {zero_stats['std_perimeter']:.1f}\n")
        f.write(f"• Average area: {zero_stats['avg_area']:.1f} ± {zero_stats['std_area']:.1f}\n")
        
        # Find optimal thresholds
        next_lowest_edges = min([digit_averages[d]['avg_edges'] for d in digit_averages.keys() if d != 0])
        next_lowest_contours = min([digit_averages[d]['avg_contours'] for d in digit_averages.keys() if d != 0])
        
        f.write(f"\nOptimal thresholds for zero detection:\n")
        f.write(f"• Edge threshold: < {next_lowest_edges * 0.8:.1f} (80% of next lowest: {next_lowest_edges:.1f})\n")
        f.write(f"• Contour threshold: < {next_lowest_contours * 0.8:.1f} (80% of next lowest: {next_lowest_contours:.1f})\n")
        f.write(f"• Combined threshold: (edges < {next_lowest_edges * 0.8:.1f}) AND (contours < {next_lowest_contours * 0.8:.1f})\n")

print(f"Summary report saved to: {report_path}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print(f"Results saved to: {output_dir}/")
print("- edge_contour_analysis.csv (detailed data)")
print("- edge_contour_analysis.png (visualizations)")
print("- edge_contour_report.txt (summary report)")
print("=" * 70)
