##### Design for an audience
- **Familiarity principle** typically favors simple graphs over complicated, eye-catching graphs
- Matplotlib has two interfaces:
    - A functional interface: we use functions to create and modify plots. We used this approach when we called the function eg. plt.barh()
    - An object-oriented (OO) interface: we use methods to create and modify plots. It offers more power and flexibility in graph editing
- plt.subplots(): create a graph using the OO interface
    - assign the two objects inside the tuple to variables fig and ax
    - The matplotlib.figure.Figure object acts as a canvas on which we can add one or more plots
    - The matplotlib.axes._subplots.AxesSubplot object is the actual plot
    - To create a bar plot, we use the Axes.bar() method and call plt.show() to :
    ```
    fig, ax = plt.subplots()
    ax.bar(['A', 'B', 'C'],
       [2, 4, 16])
    ```
-  Design principles help us in two ways:
    - Generate design options.
    - Help us choose among those options.
- Generally, a graph has three elements:
    - Data elements: the numbers and the categories visually represented and the relationships between them.
    - Structural elements: the axes, the ticks, the legend, the grid, etc.
    - Decorations: extra colors, shapes, artistic drawings etc.
- Maximizing the data elements ensures the audience's attention is on the data
- *The Visual Display of Quantitative Information (1983)* (Edward Tufte) 
- Example: [top20_deathtoll.jpg]
    - Identifying the data-ink that we can't remove without losing information (core of the graph):
        - The bars
        - The y-tick labels (the country names)
        - The x-tick labels (the number of deaths)
    - Two structural elements that constitute non-data ink (can be removed): the axes and the ticks
        - To remove the axes (spines): `Axes.spines[position].set_visible(bool)` (position is a string indicating the location of the axis - left/right/top/bottom)
        - To remove all spines:
        ```
        for location in ['left', 'right', 'bottom', 'top']:
        ax.spines[location].set_visible(False)
        ```
        - To remove left and bottom ticks: `ax.tick_params(bottom=False, left=False)`
    - Erase redundant data-ink:
        - Make the bars less thick and remove some of the x-tick labels:
        ```python
        ax.barh(top20_deathtoll['Country_Other'],
        top20_deathtoll['Total_Deaths'],
        height=0.1)
        ax.set_xticks([0, 100000, 200000, 300000])
        ```
- Consider audience's direction of reading: move the tick labels at the top: 
```
ax.xaxis.tick_top()
ax.tick_params(top=False, left=False)
```
- To make readers to focus on the data:
    - Color the x-tick labels in grey so they don't stand out visually so much: `ax.tick_params(axis='x', colors='grey')`
    - Color the bars in a shade of red.
- HEX color codes: https://www.color-hex.com/
- Add a subtitle that explains what the quantity describes and when the data was collected.
- Use the title to show readers more data — we'll report that the death toll worldwide has surpassed 1.5M
- Generally, the title must be data ink. If we need to give structural explanations in text, we can use the subtitle. That's because the title is always so noticeable, and we need to leverage that to show more data (and also maximize the data-ink ratio).
- Axes.text(): add a title and a subtitle with 3 arguments:
    - x and y: the coordinates that give the position of the text.
    - s: the text.
- For our graph, we're not going to center the text. Instead, we're going to left-align it on the same line with the South Africa labe
- Make the y-tick labels easier to read: add thousand commas: `ax.set_xticklabels(['0', '150,000', '300,000'])`
- Left-align the y-tick labels (the country names):
    - Remove the current labels: `ax.set_yticklabels([])`
    - Re add the labels with new positions:
    ```python
    for i, country in zip(range(0, 21), country_names):
    ax.text(x=-80000, y=i-0.15, s=country)
    ```
- Add a vertical line below 150,000:
    - set ymin parameter: 0 is bottom, 1 is top of the plot
    - c parameter to change the color
    - alpha parameter: transparency
    - `ax.axvline(x=150000, ymin=0.045, c='grey', alpha=0.5)`

##### Storytelling data visualization
- To create a data story, we need to wrap those numerical facts into events that show change.
- Axes.axhline(): draw a horizontal line. y parameter specifies the y-coordinate of the horizontal line.
