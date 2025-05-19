import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_figure(
        image_arrays,
        image_types,
        figure_title=None,
        figure_subtitle=None
):
    if not isinstance(image_arrays, list):
        image_arrays = [image_arrays]
    padding = (1 - image_arrays[0].shape[-1] / image_arrays[0].shape[-2]) / 2
    figure = make_subplots(
        rows=1,
        cols=len(image_arrays),
        subplot_titles=image_types,
        specs=[[
            {'l': padding, 'r': padding} for _ in range(len(image_arrays))
        ]]
    )
    if len(image_arrays[0].shape) == 3:
        for column_index, (image_array, image_type) in enumerate(zip(image_arrays, image_types), 1):
            if (
                    image_type == "Image"
                    or image_type == "Weakly augmented image"
                    or image_type == "Strongly augmented image"
            ):
                for i in range(image_array.shape[0]):
                    figure.add_trace(
                        go.Heatmap(
                            z=image_array[i],
                            zmid=0.0,
                            colorscale='gray',
                            showscale=False,
                            name="",
                            visible=True if i == 0 else False
                        ),
                        row=1,
                        col=column_index
                    )

    figure_title = (
        f"<span style='font-size:30'>{figure_title}</span>"
        if figure_title else ""
    )
    figure_subtitle = (
        f"<span style='font-size:16'>{figure_subtitle}</span>"
        if figure_subtitle else ""
    )
    figure.update_layout(
        height=500,
        width=950 if len(image_arrays) == 3 else 750,
        margin=dict(t=175),
        title=dict(
            text=f"{figure_title}<br><br>{figure_subtitle}",
            font=dict(size=16),
            x=0.5,
            y=0.9,
            xanchor="center",
            yanchor="top"
        ),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        xaxis2=dict(showgrid=False, showticklabels=False),
        yaxis2=dict(showgrid=False, showticklabels=False),
        xaxis3=dict(showgrid=False, showticklabels=False),
        yaxis3=dict(showgrid=False, showticklabels=False)
    )
    figure.update_annotations(font_size=16, yshift=10)

    if len(image_arrays[0].shape) == 3:
        steps = []
        for slice_index, i in enumerate(range(0, image_arrays[0].shape[0])):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(figure.data)}],
                label=slice_index
            )
            for j in range(i, len(figure.data), image_arrays[0].shape[0]):
                step["args"][0]["visible"][j] = True
            steps.append(step)
        figure.update_layout(
            sliders=[dict(
                active=0,
                currentvalue=dict(
                    prefix="Slice: ",
                    font=dict(size=16)
                ),
                font=dict(size=16),
                len=0.75,
                pad={"t": 25},
                steps=steps,
                x=0.125
            )]
        )
    figure.show()

def show_preprocessed_data_figure(
        image_array,
        mask_array=None,
        data_dimension=None,
        separated_images=False,
        figure_title=None,
        figure_subtitle=None
):
    if data_dimension in {"2D", "2.5D", "3D"}:
        if separated_images:
            padding = (1 - image_array.shape[-1] / image_array.shape[-2]) / 2
            figure = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=["Image", "Mask", "Image + mask"],
                specs=[[
                    {'l': padding, 'r': padding},
                    {'l': padding, 'r': padding},
                    {'l': padding, 'r': padding}
                ]]
            )
            if len(image_array.shape) == 3:
                for i in range(image_array.shape[0]):
                    figure.add_trace(
                        go.Heatmap(
                            z=image_array[i],
                            zmid=0.5,
                            colorscale='gray',
                            showscale=False,
                            name="",
                            visible=True if i == 0 else False
                        ),
                        row=1,
                        col=1
                    )
                    figure.add_trace(
                        go.Heatmap(
                            z=image_array[i],
                            zmid=0.5,
                            colorscale='gray',
                            showscale=False,
                            name="",
                            visible=True if i == 0 else False
                        ),
                        row=1,
                        col=3
                    )
                    if mask_array is not None:
                        figure.add_trace(
                            go.Heatmap(
                                z=mask_array[i],
                                zmid=0.,
                                colorscale=[
                                    [0, "black"],
                                    [0.5, "black"],
                                    [1, "red"]
                                ],
                                showscale=False, name="",
                                visible=True if i == 0 else False
                            ),
                            row=1, col=2)
                        figure.add_trace(
                            go.Heatmap(
                                z=mask_array[i], zmid=0.5,
                                colorscale=[
                                    [0, "rgba(0,0,0,0)"],
                                    [0.5, "rgba(0,0,0,0)"],
                                    [1, "red"]
                                ],
                                showscale=False,
                                opacity=0.3,
                                name="",
                                visible=True if i == 0 else False
                            ),
                            row=1,
                            col=3
                        )
            else:
                figure.add_trace(
                    go.Heatmap(
                        z=image_array,
                        zmid=0.5,
                        colorscale='gray',
                        showscale=False,
                        name=""
                    ),
                    row=1,
                    col=1
                )
                figure.add_trace(
                    go.Heatmap(
                        z=image_array,
                        zmid=0.5,
                        colorscale='gray',
                        showscale=False,
                        name=""
                    ),
                    row=1,
                    col=3
                )
                if mask_array is not None:
                    figure.add_trace(
                        go.Heatmap(
                            z=mask_array,
                            zmid=0.5,
                            colorscale=[
                                [0, "black"],
                                [0.5, "black"],
                                [1, "red"]
                            ],
                            showscale=False,
                            name=""
                        ),
                        row=1,
                        col=2
                    )
                    figure.add_trace(
                        go.Heatmap(
                            z=mask_array,
                            zmid=0.5,
                            colorscale=[
                                [0, "rgba(0,0,0,1)"],
                                [0.5, "rgba(0,0,0,1)"],
                                [1, "red"]
                            ],
                            showscale=False,
                            opacity=0.3,
                            name=""
                        ),
                        row=1,
                        col=3
                    )

            figure_title = (
                f"<span style='font-size:30'>{figure_title}</span>"
                if figure_title else ""
            )
            figure_subtitle = (
                f"<span style='font-size:16'>{figure_subtitle}</span>"
                if figure_subtitle else ""
            )
            figure.update_layout(
                height=500,
                width=950,
                margin=dict(t=175),
                title=dict(
                    text=f"{figure_title}<br><br>{figure_subtitle}",
                    font=dict(size=16),
                    x=0.5,
                    y=0.9,
                    xanchor="center",
                    yanchor="top"
                ),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                xaxis2=dict(showgrid=False, showticklabels=False),
                yaxis2=dict(showgrid=False, showticklabels=False),
                xaxis3=dict(showgrid=False, showticklabels=False),
                yaxis3=dict(showgrid=False, showticklabels=False)
            )
            figure.update_annotations(font_size=16, yshift=10)

            if len(image_array.shape) == 3:
                steps = []
                for slice_index, i in enumerate(range(0, len(figure.data), 4)):
                    step = dict(
                        method="update",
                        args=[{"visible": [False] * len(figure.data)}],
                        label=slice_index
                    )
                    for j in range(i, i + 4):
                        step["args"][0]["visible"][j] = True
                    steps.append(step)
                figure.update_layout(
                    sliders=[dict(
                        active=0,
                        currentvalue=dict(
                            prefix="Slice: ",
                            font=dict(size=16)
                        ),
                        font=dict(size=16),
                        len=0.75,
                        pad={"t": 25},
                        steps=steps,
                        x=0.125
                    )]
                )
        else:
            padding = (1 - image_array.shape[-1] / image_array.shape[-2]) / 2
            figure = make_subplots(
                rows=1,
                cols=1,
                specs=[[{'l': padding, 'r': padding}]]
            )
            if len(image_array.shape) == 3:
                for i in range(image_array.shape[0]):
                    figure.add_trace(
                        go.Heatmap(
                            z=image_array[i],
                            zmid=0.5,
                            colorscale='gray',
                            showscale=False,
                            name="",
                            visible=True if i == 0 else False
                        ),
                        row=1,
                        col=1
                    )
                    if mask_array is not None:
                        figure.add_trace(
                            go.Heatmap(
                                z=mask_array[i],
                                zmid=0.5,
                                colorscale=[
                                    [0, "rgba(0,0,0,0)"],
                                    [0.5, "rgba(0,0,0,0)"],
                                    [1, "red"]
                                ],
                                showscale=False,
                                opacity=0.3,
                                name="",
                                visible=True if i == 0 else False
                            ),
                            row=1,
                            col=1
                        )
            else:
                figure.add_trace(
                    go.Heatmap(
                        z=image_array,
                        zmid=0.5,
                        colorscale='gray',
                        showscale=False,
                        name=""
                    ),
                    row=1,
                    col=1
                )
                if mask_array is not None:
                    figure.add_trace(go.Heatmap(
                        z=mask_array, zmid=0.5,
                        colorscale=[
                            [0, "rgba(0,0,0,0)"],
                            [0.5, "rgba(0,0,0,0)"],
                            [1, "red"]
                        ],
                        showscale=False, opacity=0.3, name=""), row=1, col=1)

            figure_title = (
                f"<span style='font-size:30'>{figure_title}</span>"
                if figure_title else ""
            )
            figure_subtitle = (
                f"<span style='font-size:16'>{figure_subtitle}</span>"
                if figure_subtitle else ""
            )
            figure.update_layout(
                height=500,
                width=450,
                margin=dict(t=125),
                title=dict(
                    text=f"{figure_title}<br><br>{figure_subtitle}",
                    font=dict(size=16),
                    x=0.5,
                    y=0.9,
                    xanchor="center",
                    yanchor="top"
                ),
                xaxis=dict(showgrid=False,  showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False)
            )

            if len(image_array.shape) == 3:
                steps = []
                for slice_index, i in enumerate(range(0, len(figure.data), 2)):
                    step = dict(
                        method="update",
                        args=[{"visible": [False] * len(figure.data)}],
                        label=slice_index
                    )
                    for j in range(i, i + 2):
                        step["args"][0]["visible"][j] = True
                    steps.append(step)
                figure.update_layout(
                    sliders=[
                        dict(
                            active=0,
                            currentvalue={"prefix": "Slice: "},
                            pad={"t": 25},
                            steps=steps)
                    ]
                )
    else:
        raise ValueError("Invalid data dimension")
    figure.show()

def show_raw_data_figure(
        image_array,
        figure_title=None,
        figure_subtitle=None,
):
    figure = make_subplots(rows=1, cols=1)
    for i in range(image_array.shape[0]):
        figure.add_trace(
            go.Heatmap(
                z=image_array[i],
                zmid=0.5,
                colorscale='gray',
                showscale=False,
                name="",
                visible=True if i == 0 else False
            ),
            row=1,
            col=1
        )

    figure_title = (
        f"<span style='font-size:30'>{figure_title}</span>"
        if figure_title else ""
    )
    figure_subtitle = (
        f"<span style='font-size:16'>{figure_subtitle}</span>"
        if figure_subtitle else ""
    )
    figure.update_layout(
        height=500,
        width=450,
        margin=dict(t=125),
        title=dict(
            text=f"{figure_title}<br><br>{figure_subtitle}",
            font=dict(size=16),
            x=0.5,
            y=0.9,
            xanchor="center",
            yanchor="top"
        ),
        xaxis=dict(showgrid=False,  showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )

    steps = []
    for slice_index, i in enumerate(range(0, len(figure.data), 1)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(figure.data)}],
            label=slice_index
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)
    figure.update_layout(
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Slice: "},
                pad={"t": 25},
                steps=steps)
        ]
    )
    figure.show()