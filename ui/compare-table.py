import os
import numpy as np
import requests
import pandas as pd
import re
import time
from collections import Counter

def generate_summary(dataframe: pd.DataFrame):
    pass

def generate_comparison_table(dataframe: pd.DataFrame, stats = ''):
    table_html = dataframe.to_html(index=False, border=0, table_id="table", classes='table table-striped', justify='left', escape=False, render_links=True).replace('\\n', '<br>')
    # https://codepen.io/mugunthan/pen/RwbVqYO https://mark-rolich.github.io/Magnifier.js/ https://github.com/malaman/js-image-zoom
    script = """
    <script src="https://code.jquery.com/jquery-3.6.4.slim.min.js" integrity="sha256-a2yjHM4jnF9f54xUQakjZGaqYs/V1CYvWpoqZzC2/Bw=" crossorigin="anonymous"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"> type="text/javascript"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"> type="text/javascript"></script>
    <script>
        var options1 = {
            width: 300,
            height: 400,
            zoomWidth: 300,
            offset: {vertical: 0, horizontal: 10}
        };
        const containers = document.querySelectorAll('.img-container');
        containers.forEach((container) => {
            new ImageZoom(container, options1);
        });
        $(document).ready(function () {
            $('#table').DataTable(
                dom: 'Bfrtip',
                buttons: [
                    {
                        extend: 'colvis',
                        text: 'Toggle columns'
                    }
                ]
            ); 
        });
    </script>
    """

    return f"""
    <html>
    <head>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
        <script src="https://unpkg.com/js-image-zoom@0.7.0/js-image-zoom.js" type="application/javascript"></script>
        <link href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css" rel="stylesheet">
        <style>
        .img-container {{
            max-width: 300px;
            max-height: 400px;
            display: block;
        }}
        .img-container-wrapper {{ width: 550px; display: block;}}
        tbody tr {{ height: 400px; }}
        body {{ padding: 2em; }}
        .common {{ background-color: #ddffd6; padding: 2px; }}
        h4 {{ font-size: 0.8em; font-weight: bold;  }}
        .different {{ background-color: #ffd6d6; padding: 2px; }}
        #stats {{ padding: 15px 0 }}
        </style>
    </head>
    <body>
    <div id="stats">{stats}</div>
    {table_html}
    {script}
    </body>
    </html>
    """

def image_html(img_url):
    return f'<div class="img-container-wrapper"><div class="img-container"><img src="{img_url}" class="zoom-image"></div></div>'

# def map_dicts_to_keys(dict1, dict2, keys):
#     return ({key: dict1.get(key, None) for key in keys},
#             {key: dict2.get(key, None) for key in keys})

# def sort_dict(dct):
#     return dict(sorted(dct.items()))
