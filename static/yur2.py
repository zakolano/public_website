import re

def parse_and_sort_css_coordinates_from_file(filepath):
    # Read the CSS content from the file
    with open(filepath, 'r') as file:
        css_content = file.read()
    
    # Regular expression to find elements starting with _91 or _52 and extract their left and top values
    pattern = re.compile(r'\.(?:_97)[^ ]* {[^}]*left: (\d+)px;[^}]*top: (\d+)px;')
    matches = pattern.findall(css_content)
    
    # Convert matches to a list of tuples with integer values
    coordinates = [(int(left), int(top)) for left, top in matches]
    
    # Remove duplicates
    coordinates = list(set(coordinates))
    
    # Define the order of left values
    left_order = [186, 1148, 419, 910, 568, 761 , 554, 779, 560, 799, 674]
    
    # Sort coordinates by the specified left order and then by top value
    sorted_coordinates = sorted(coordinates, key=lambda x: (left_order.index(x[0]) if x[0] in left_order else len(left_order), x[1]))
    
    # Convert sorted coordinates to a list of dictionaries with 'px' appended
    formatted_coordinates = [{'left': f'{left}', 'top': f'{top}'} for left, top in sorted_coordinates]
    
    return formatted_coordinates

# Example usage
filepath = 'c:/Users/zakol/keep/zakol_folder/website_maybe/static/asdas.css'
coordinates = parse_and_sort_css_coordinates_from_file(filepath)

# Print the formatted coordinates in a format that can be saved to a file
print("position_dict = [")
for coord in coordinates:
    print(f"    {{'left': {coord['left']}, 'top': {coord['top']}}},")
print("]")