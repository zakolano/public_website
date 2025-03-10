import re

# Define the file paths
css_file_path = 'c:/Users/zakol/keep/zakol_folder/website_maybe/static/asdas.css'
html_file_path = 'c:/Users/zakol/keep/zakol_folder/website_maybe/templates/asdas.html'

# Function to update CSS file based on HTML
def update_css_based_on_html(css_file_path, html_file_path):
    # Read the HTML file
    with open(html_file_path, 'r') as html_file:
        html_content = html_file.read()

    # Find all SVG elements with width and height attributes
    svg_elements = re.findall(r'<svg class="(line-\d+)" width="(\d+)" height="(\d+)"', html_content)

    # Read the CSS file
    with open(css_file_path, 'r') as css_file:
        css_content = css_file.read()

    # Update the CSS rules based on the SVG elements
    for svg_class, width, height in svg_elements:
        width = int(width)
        height = int(height)
        if height > width:
            print(height, width, svg_class)

            # Update width in CSS
            css_content = re.sub(
                rf'(\.{svg_class}\s*{{[^}}]*width:\s*)\d+px;',
                lambda m: f"{m.group(1)}{width}px;",
                css_content
            )

            # Update height in CSS
            css_content = re.sub(
                rf'(\.{svg_class}\s*{{[^}}]*height:\s*)\d+px;',
                lambda m: f"{m.group(1)}{height}px;",
                css_content
            )

    # Write the updated CSS content back to the file
    with open(css_file_path, 'w') as css_file:
        css_file.write(css_content)

    print("CSS file has been updated based on the HTML file.")

# Update the CSS file based on the HTML file
update_css_based_on_html(css_file_path, html_file_path)
