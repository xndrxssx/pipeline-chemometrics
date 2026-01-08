import pandas as pd
import json
import base64
from datetime import datetime

def generate_interactive_report(results, output_path):
    """
    Generates a premium HTML dashboard with summary cards, advanced tables, and detailed modals.
    """
    
    df = pd.DataFrame(results)
    
    # 1. Calculate Summary Stats
    total_models = len(df)
    best_reg = df[df['Task'] == 'Regression']['R2_Test'].max() if not df[df['Task'] == 'Regression'].empty else 0
    best_cls = df[df['Task'] == 'Classification']['Accuracy'].max() if not df[df['Task'] == 'Classification'].empty else 0
    avg_time = df['Training_Time'].mean() if 'Training_Time' in df.columns else 0
    
    # 2. Prepare Data for JS
    # Round numeric columns for display
    numeric_cols = ['R2_Test', 'RMSE_Test', 'RPD', 'Accuracy', 'Training_Time', 'Bias', 'Slope', 'Offset']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert to list of dicts
    table_data = df.drop(columns=['Plot_Base64', 'Model_Obj'], errors='ignore').fillna("-").to_dict(orient='records')
    
    # Store Base64 plots separately
    plots_data = {}
    if 'Plot_Base64' in df.columns:
        for idx, row in df.iterrows():
            if row['Plot_Base64']:
                plots_data[idx] = row['Plot_Base64']

    json_table = json.dumps(table_data)
    json_plots = json.dumps(plots_data)
    
    # 3. HTML Template
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chemometrics Dashboard</title>
        
        <!-- Fonts & Icons -->
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
        
        <!-- Bootstrap 5 & DataTables -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css" rel="stylesheet">
        <link href="https://cdn.datatables.net/buttons/2.2.2/css/buttons.bootstrap5.min.css" rel="stylesheet">
        
        <style>
            :root {{ --primary-color: #4361ee; --secondary-color: #3f37c9; --bg-color: #f8f9fa; }}
            body {{ font-family: 'Inter', sans-serif; background-color: var(--bg-color); color: #2b2d42; }}
            
            /* Cards */
            .stats-card {{ border: none; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.2s; }}
            .stats-card:hover {{ transform: translateY(-5px); }}
            .icon-box {{ width: 48px; height: 48px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; }}
            
            /* Table */
            .card-table {{ border-radius: 12px; border: none; box-shadow: 0 4px 12px rgba(0,0,0,0.05); overflow: hidden; }}
            table.dataTable thead th {{ background-color: #e9ecef; color: #495057; font-weight: 600; border-bottom: 2px solid #dee2e6; }}
            
            /* Progress Bars */
            .progress {{ height: 8px; border-radius: 4px; background-color: #e9ecef; margin-top: 6px; }}
            
            /* Modal */
            .modal-content {{ border-radius: 15px; border: none; }}
            .modal-header {{ background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); color: white; border-radius: 15px 15px 0 0; }}
            .params-box {{ background: #f1f3f5; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.85rem; max-height: 200px; overflow-y: auto; }}
        </style>
    </head>
    <body>
        
        <!-- Navbar -->
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4 shadow-sm">
            <div class="container-fluid">
                <a class="navbar-brand" href="#"><i class="bi bi-bar-chart-fill me-2"></i>Chemometrics Pipeline</a>
                <span class="text-white-50 ms-auto small">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
            </div>
        </nav>

        <div class="container-fluid px-4">
            
            <!-- Summary Cards -->
            <div class="row g-4 mb-4">
                <div class="col-md-3">
                    <div class="card stats-card h-100">
                        <div class="card-body d-flex align-items-center">
                            <div class="icon-box bg-primary text-white me-3"><i class="bi bi-hdd-stack"></i></div>
                            <div>
                                <h6 class="text-muted mb-1">Total Models</h6>
                                <h4 class="mb-0 fw-bold">{total_models}</h4>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stats-card h-100">
                        <div class="card-body d-flex align-items-center">
                            <div class="icon-box bg-success text-white me-3"><i class="bi bi-graph-up-arrow"></i></div>
                            <div>
                                <h6 class="text-muted mb-1">Best Regression R²</h6>
                                <h4 class="mb-0 fw-bold">{best_reg:.4f}</h4>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stats-card h-100">
                        <div class="card-body d-flex align-items-center">
                            <div class="icon-box bg-info text-white me-3"><i class="bi bi-bullseye"></i></div>
                            <div>
                                <h6 class="text-muted mb-1">Best Accuracy</h6>
                                <h4 class="mb-0 fw-bold">{best_cls*100:.1f}%</h4>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stats-card h-100">
                        <div class="card-body d-flex align-items-center">
                            <div class="icon-box bg-warning text-white me-3"><i class="bi bi-stopwatch"></i></div>
                            <div>
                                <h6 class="text-muted mb-1">Avg Train Time</h6>
                                <h4 class="mb-0 fw-bold">{avg_time:.3f}s</h4>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Table -->
            <div class="card card-table mb-5">
                <div class="card-body p-4">
                    <h5 class="card-title mb-4">Model Registry</h5>
                    
                    <!-- Custom Filters -->
                    <div class="row mb-3 g-2">
                        <div class="col-md-3">
                            <select id="filterDataset" class="form-select form-select-sm"><option value="">All Datasets</option></select>
                        </div>
                        <div class="col-md-3">
                            <select id="filterTarget" class="form-select form-select-sm"><option value="">All Targets</option></select>
                        </div>
                        <div class="col-md-3">
                            <select id="filterModel" class="form-select form-select-sm"><option value="">All Models</option></select>
                        </div>
                    </div>

                    <table id="mainTable" class="table table-hover w-100">
                        <thead>
                            <tr>
                                <th>Dataset</th>
                                <th>Target</th>
                                <th>Preproc</th>
                                <th>Model</th>
                                <th style="width: 120px;">Score (R²/Acc)</th>
                                <th>RMSE</th>
                                <th>RPD</th>
                                <th>Time (s)</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Detail Modal -->
        <div class="modal fade" id="detailModal" tabindex="-1">
            <div class="modal-dialog modal-lg modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title"><i class="bi bi-clipboard-data me-2"></i>Model Details</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        
                        <ul class="nav nav-tabs mb-3" id="modalTabs" role="tablist">
                            <li class="nav-item">
                                <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-plot">Plot</button>
                            </li>
                            <li class="nav-item">
                                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-metrics">Metrics</button>
                            </li>
                            <li class="nav-item">
                                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-params">Config</button>
                            </li>
                        </ul>
                        
                        <div class="tab-content">
                            <!-- Plot Tab -->
                            <div class="tab-pane fade show active text-center" id="tab-plot">
                                <img id="modalPlot" src="" class="img-fluid rounded shadow-sm" alt="No Plot Available">
                                <div id="noPlotMsg" class="alert alert-warning mt-3 d-none">No plot generated for this model.</div>
                            </div>
                            
                            <!-- Metrics Tab -->
                            <div class="tab-pane fade" id="tab-metrics">
                                <table class="table table-sm table-striped">
                                    <tbody id="metricsTableBody">
                                        <!-- JS populates this -->
                                    </tbody>
                                </table>
                            </div>
                            
                            <!-- Params Tab -->
                            <div class="tab-pane fade" id="tab-params">
                                <div id="modalParams" class="params-box"></div>
                            </div>
                        </div>

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
                
                // Initialize DataTable
                const table = $('#mainTable').DataTable({{
                    data: tableData,
                    columns: [
                        {{ data: 'dataset' }},
                        {{ data: 'target', render: function(data) {{ return `<span class="badge bg-light text-dark border">${{data}}</span>`; }} }},
                        {{ data: 'preprocessing', render: function(data) {{ return `<small>${{data}}</small>`; }} }},
                        {{ data: 'model', className: 'fw-bold' }},
                        {{ 
                            data: null, 
                            render: function(data, type, row) {{
                                // Choose R2 or Accuracy
                                let val = row.Task === 'Classification' ? row.Accuracy : row.R2_Test;
                                let color = val > 0.8 ? 'success' : (val > 0.5 ? 'warning' : 'danger');
                                let width = Math.max(0, Math.min(100, val * 100));
                                return `
                                    <div class="d-flex justify-content-between small mb-1">
                                        <span>${{val.toFixed(4)}}</span>
                                    </div>
                                    <div class="progress" style="height: 6px;">
                                        <div class="progress-bar bg-${{color}}" style="width: ${{width}}%"></div>
                                    </div>
                                `;
                            }}
                        }},
                        {{ data: 'RMSE_Test', render: $.fn.dataTable.render.number(',', '.', 4) }},
                        {{ data: 'RPD', render: function(data) {{ return data ? parseFloat(data).toFixed(2) : '-'; }} }},
                        {{ data: 'Training_Time', render: function(data) {{ return data ? parseFloat(data).toFixed(3) : '-'; }} }},
                        {{ 
                            data: null, 
                            orderable: false,
                            render: function(data, type, row, meta) {{
                                return `<button class="btn btn-sm btn-outline-primary" onclick="openModal(${{meta.row}})">
                                    <i class="bi bi-eye"></i> Details
                                </button>`;
                            }}
                        }}
                    ],
                    order: [[ 4, "desc" ]],
                    dom: 'tp', // Custom layout
                    pageLength: 20
                }});

                // Populate Custom Dropdowns
                function populateFilter(id, columnIdx) {{
                    var column = table.column(columnIdx);
                    var select = $(id);
                    column.data().unique().sort().each(function(d, j) {{
                        if(d) select.append('<option value="'+d+'">'+d+'</option>')
                    }});
                    select.on('change', function() {{
                        var val = $.fn.dataTable.util.escapeRegex($(this).val());
                        column.search(val ? '^'+val+'$' : '', true, false).draw();
                    }});
                }}
                
                populateFilter('#filterDataset', 0); // Dataset col
                # Target col has HTML badges, so filtering is trickier. 
                # Simplified: filtering by raw data would require re-rendering logic.
                # For now let's use the search box for target or adjust column def.
                # Actually, DataTables orthogonal data handles this, but let's stick to standard search for Target.
                
                // Re-implement simplified filter logic for Target and Model
                ['dataset', 'target', 'model'].forEach(key => {{
                    const unique = [...new Set(tableData.map(item => item[key]))].sort();
                    const selector = '#filter' + key.charAt(0).toUpperCase() + key.slice(1);
                    if($(selector).length) {{
                        unique.forEach(u => $(selector).append(`<option value="${{u}}">${{u}}</option>`));
                        $(selector).on('change', function() {{
                            table.column(key+':name').search(this.value).draw();
                        }});
                    }}
                }});
            }});

            function openModal(rowIndex) {{
                const row = tableData[rowIndex];
                const plot = plotsData[rowIndex];
                
                // 1. Plot Tab
                const img = $('#modalPlot');
                const msg = $('#noPlotMsg');
                if (plot) {{
                    img.attr('src', 'data:image/png;base64,' + plot).removeClass('d-none');
                    msg.addClass('d-none');
                }} else {{
                    img.addClass('d-none');
                    msg.removeClass('d-none');
                }}
                
                // 2. Metrics Tab
                let rows = '';
                const exclude = ['Plot_Base64', 'params', 'Model_Obj'];
                for (const [key, val] of Object.entries(row)) {{
                    if (!exclude.includes(key) && val !== null && val !== "") {{
                        rows += `<tr><th class="text-muted w-50">${{key}}</th><td class="fw-bold">${{val}}</td></tr>`;
                    }}
                }}
                $('#metricsTableBody').html(rows);
                
                // 3. Params Tab
                try {{
                    // Try to pretty print JSON params
                    const pObj = eval('(' + row.params + ')');
                    $('#modalParams').text(JSON.stringify(pObj, null, 2));
                }} catch(e) {{
                    $('#modalParams').text(row.params);
                }}
                
                new bootstrap.Modal(document.getElementById('detailModal')).show();
            }}
        </script>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"📄 Premium Interactive Report generated: {output_path}")
