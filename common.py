from flask import Flask

# Common styling for HTML pages
def get_common_styles():
    return """
    <style>
        body { font-family: Arial, sans-serif; }
        .container { display: flex; }
        .file-list { width: 50%; }
        .image-viewer { width: 50%; text-align: center; }
        img { max-width: 100%; height: auto; max-height: 300px; }
        ul { list-style-type: none; padding: 0; }
        li { margin-bottom: 10px; }
        a { text-decoration: none; color: blue; }
        a:hover { text-decoration: underline; }
        .selected { font-weight: bold; color: red; }
        pre { text-align: left; overflow-x: auto; background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; }

        /* Updated navbar styles */
        nav {
            background-color: #333;
            padding: 10px 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        nav ul {
            list-style: none;
            display: flex;
            margin: 0;
            padding: 0;
        }
        nav li {
            margin-right: 20px;
        }
        nav a {
            color: white;
            text-decoration: none;
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        nav a:hover {
            background-color: #555;
        }
        nav a[style*="font-weight: bold;"] {
            background-color: #555;
        }
    </style>
    """

def add_script():
    return """<script>
        function showImage(timestamp, filename, folder) {
            const viewer = document.getElementById('image-viewer');
            const links = document.querySelectorAll('#file-list a');
            links.forEach(link => link.classList.remove('selected'));
            document.getElementById(`link-${timestamp}`).classList.add('selected');

            viewer.innerHTML = `
                <h2>Image details</h2>
                <img src="/${folder}/${timestamp}_${filename}.jpg" alt="Selected Image" style="max-height: 300px;">
                <img src="/${folder}/${timestamp}_roi_${filename}.jpg" alt="ROI Image">
                <img src="/${folder}/${timestamp}_box_${filename}.jpg" alt="Box Image">
            `;
                const jsonFile = timestamp.replace('.jpg', '.json');
                fetch(`/${folder}/${timestamp}_${filename}.json`)
                    .then(response => response.json())
                    .then(data => {
                        viewer.innerHTML += `
                            <h3>JSON Information</h3>
                            <pre>${JSON.stringify(data, null, 2)}</pre>
                        `;
                    })
                    .catch(error => {
                        viewer.innerHTML += `<p style=\"color:red;\">No JSON</p>`;
                    });
        }
    </script>"""

# Common navigation bar
def generate_navbar(active_page):
    nav_items = [
        {"name": "Processed Images", "url": "/files", "id": "files"},
        {"name": "Wrongly Classified", "url": "/wrong", "id": "wrong"},
        {"name": "Correctly Classified", "url": "/correct", "id": "correct"},
        {"name": "View Log", "url": "/log", "id": "log"},
    ]

    navbar = '<nav><ul style="list-style: none; display: flex; padding: 0;">'
    for item in nav_items:
        active_class = 'style="font-weight: bold;"' if item["id"] == active_page else ''
        navbar += f'<li style="margin-right: 15px;"><a href="{item["url"]}" {active_class}>{item["name"]}</a></li>'
    navbar += '</ul></nav>'
    return navbar

# Common HTML structure
def generate_html(active_page, content):
    return f"""
    <html>
    <head>
        {get_common_styles()}
        {add_script()}
    </head>
    <body>
        {generate_navbar(active_page)}
        {content}
    </body>
    </html>
    """