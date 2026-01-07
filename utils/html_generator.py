import pandas as pd
import json
import base64
from datetime import datetime

def generate_interactive_report(results, output_path):
    """
    Generates a single-file HTML report with interactive tables and embedded charts.
    """
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Helper: Convert numeric columns to rounded strings for display
    numeric_cols = ['R2_Test', 'RMSE_Test', 'RPD', 'Accuracy', 'Training_Time']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else "-")

    # Serialize data to JSON for JS handling
    table_data = df.drop(columns=['Plot_Base64', 'Model_Obj'], errors='ignore').to_dict(orient='records')
    plots_data = {}
    
    # Store Base64 plots in a separate JS object keyed by a unique ID
    if 'Plot_Base64' in df.columns:
        for idx, row in df.iterrows():
            if row['Plot_Base64']:
                plots_data[idx] = row['Plot_Base64']
                
    json_table = json.dumps(table_data)
    json_plots = json.dumps(plots_data)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chemometrics Pipeline Report</title>
        
        <!-- Bootstrap & DataTables CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css" rel="stylesheet">
        <link href="https://cdn.datatables.net/buttons/2.2.2/css/buttons.bootstrap5.min.css" rel="stylesheet">
        
        <style>
            body {{ background-color: #f8f9fa; font-size: 0.9rem; }}
            .container-fluid {{ padding: 20px; }}
            .modal-dialog {{ max_width: 800px; }}
            #mainTable td {{ vertical-align: middle; }}
            .status-good {{ color: green; font-weight: bold; }}
            .status-bad {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h2 class="mb-4">🔬 Chemometrics Pipeline Report</h2>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="card shadow">
                <div class="card-body">
                    <table id="mainTable" class="table table-striped table-hover" style="width:100%">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Dataset</th>
                                <th>Target</th>
                                <th>Preprocessing</th>
                                <th>Model</th>
                                <th>R² (Test)</th>
                                <th>RMSE</th>
                                <th>RPD</th>
                                <th>Acc</th>
                                <th>Details</th>
                            </tr>
                        </thead>
                        <tbody>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Modal for Plots -->
        <div class="modal fade" id="plotModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Model Performance</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-center">
                        <img id="modalPlot" src="" class="img-fluid" alt="No Plot Available">
                        <hr>
                        <p id="modalParams" class="text-muted text-start font-monospace small"></p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Scripts -->
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.5/js/dataTables.bootstrap5.min.js"></script>
        <script src="https://cdn.datatables.net/buttons/2.2.2/js/dataTables.buttons.min.js"></script>
        <script src="https://cdn.datatables.net/buttons/2.2.2/js/buttons.html5.min.js"></script>

        <script>
            const tableData = {json_table};
            const plotsData = {json_plots};

            $(document).ready(function() {{
                const table = $('#mainTable').DataTable({{
                    data: tableData,
                    columns: [
                        {{ data: null, render: (data, type, row, meta) => meta.row }},
                        {{ data: 'dataset' }},
                        {{ data: 'target' }},
                        {{ data: 'preprocessing' }},
                        {{ data: 'model' }},
                        {{ data: 'R2_Test' }},
                        {{ data: 'RMSE_Test' }},
                        {{ data: 'RPD' }},
                        {{ data: 'Accuracy' }},
                        {{ 
                            data: null, 
                            render: function(data, type, row, meta) {{
                                const hasPlot = plotsData[meta.row] ? 'btn-primary' : 'btn-secondary disabled';
                                return `<button class="btn btn-sm ${hasPlot}" onclick="showPlot(${meta.row})">View</button>`;
                            }}
                        }}
                    ],
                    order: [[ 5, "desc" ]], // Sort by R2 by default
                    dom: 'Bfrtip',
                    buttons: ['copy', 'csv', 'excel'],
                    pageLength: 25
                }});
            }});

            function showPlot(rowIndex) {{
                const imgData = plotsData[rowIndex];
                const rowData = tableData[rowIndex];
                
                if (imgData) {{
                    $('#modalPlot').attr('src', 'data:image/png;base64,' + imgData);
                    $('#modalParams').text(JSON.stringify(eval('(' + rowData.params + ')'), null, 2));
                    new bootstrap.Modal(document.getElementById('plotModal')).show();
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"📄 Interactive Report generated: {output_path}")
