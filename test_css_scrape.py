import re

def parse_css_file(file_path):
    with open(file_path, 'r') as file:
        css_content = file.read()

    # Regular expression to find rectangle classes with their positions, widths, and heights
    rectangle_pattern = re.compile(r'\.rectangle-\d+\s*{\s*[^}]*width:\s*(\d+)px;\s*height:\s*(\d+)px;\s*position:\s*absolute;\s*left:\s*(\d+)px;\s*top:\s*(\d+)px;')

    rectangles = rectangle_pattern.findall(css_content)

    # Dictionary to store rectangles by width
    rectangles_by_width = {}

    for width, height, left, top in rectangles:
        width = int(width)
        height = int(height)
        left = int(left)
        top = int(top)
        if width not in rectangles_by_width:
            rectangles_by_width[width] = []
        rectangles_by_width[width].append((height, left, top))

    # Sort rectangles by top position
    for width in rectangles_by_width:
        rectangles_by_width[width].sort(key=lambda x: x[2])  # Sort by top

    return rectangles_by_width

def apply_left_order(rectangles_by_width, left_order, big_rectangles):
    ordered_rectangles = {}
    for width, rectangles in rectangles_by_width.items():
        ordered_rectangles[width] = []
        if width == 160:  # Apply left order only to smaller rectangles
            for left in left_order:
                filtered_rectangles = [rect for rect in rectangles if rect[1] == left]
                ordered_rectangles[width].extend(filtered_rectangles)
        else:
            ordered_rectangles[width].extend(rectangles)
    
    # Add the big rectangles at specific left positions
    for left in big_rectangles:
        ordered_rectangles[190].append((30, left, 0))  # Assuming height 30 and top 0 for big rectangles

    return ordered_rectangles

# Example usage
file_path = 'static/asdas.css'
left_order = [1, 1196, 212, 984, 387, 809, 485, 711, 376, 849]
big_rectangles = [589, 628, 591]
rectangles_by_width = parse_css_file(file_path)
ordered_rectangles = apply_left_order(rectangles_by_width, left_order, big_rectangles)

# Store the result in a list for future use
rectangles_list = []
for width, positions in ordered_rectangles.items():
    for height, left, top in positions:
        rectangles_list.append({'width': width, 'height': height, 'left': left, 'top': top})

# Print the result in a list format
print("[")
for rect in rectangles_list:
    print(f"    {{'width': {rect['width']}, 'height': {rect['height']}, 'left': {rect['left']}, 'top': {rect['top']}}},")
print("]")