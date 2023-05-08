split_index = 0
num_heads = 3
network_shape = [1, 5, 5, 2]

svg_pre = """
<svg xmlns="http://www.w3.org/2000/svg" width="1634" height="1000" style="cursor: move;">
  <g transform="translate(-147.88911403911197,-10.484958622221711) scale(1.2379902908374525)">
"""


def get_arrow_text(x1, y1, x2, y2)->str:
    return f"""<path class="link" marker-end="" d="M{x1},{y1}, {x2},{y2}"
      style="stroke-width: 0.5; stroke-opacity: 1; stroke: rgb(80, 80, 80);"></path>"""
def get_circle_text(x,y)->str:
    return f"""<circle r="6" class="node" id="{x}_{y}" cx="{x}" cy="{y}"
      style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);"></circle>"""


svg_text = svg_pre
layer_width = 120
layer_height = 500
x_value = 568 + 40
y_middle = 300 

def create_layer(x_value, y_middle, network_shape, layer_height, layer_width, i)->str:
  svg_text = ""
  # add all of the arrows
  if i != len(network_shape) - 1:
     for jj in range(network_shape[i]):
       for kk in range(network_shape[i+1]):
         svg_text += get_arrow_text(x_value, y_middle + ((jj - ((network_shape[i]-1) / 2)) / network_shape[i] * layer_height),
                                    x_value + layer_width, y_middle + ((kk - ((network_shape[i+1]-1) / 2)) / network_shape[i+1] * layer_height))
  # add all of the circles
  for j in range(network_shape[i]):
    svg_text += get_circle_text(x_value, y_middle + ((j - ((network_shape[i]-1) / 2)) / network_shape[i] * layer_height))
  return svg_text

for i in range(len(network_shape)):
  if i == split_index and i != 0:
    # need to connect the two layers
    layer_middles = [y_middle + ((j - ((num_heads-1) / 2)) /num_heads * layer_height) for j in range(num_heads)]
    for layer_middle in layer_middles:
       # create all arrows to this layer 
       for jj in range(network_shape[i]):
          for kk in range(network_shape[i+1]):
            svg_text += get_arrow_text(x_value, y_middle + ((jj - ((network_shape[i]-1) / 2)) / network_shape[i] * layer_height),
                                       x_value + layer_width, layer_middle + ((kk - ((network_shape[i+1]-1) / 2)) / network_shape[i+1] * layer_height / num_heads))
    # add all of the circles
    for j in range(network_shape[i]):
      svg_text += get_circle_text(x_value, y_middle + ((j - ((network_shape[i]-1) / 2)) / network_shape[i] * layer_height))

  elif i > split_index or split_index == 0:
     # split into separate ensembles
     for j in range(num_heads):
       # create the layer
       svg_text += create_layer(x_value, y_middle + ((j - ((num_heads-1) / 2)) /num_heads * layer_height),
                                network_shape, layer_height / num_heads, layer_width, i)

  else:
      # create the layer
      svg_text += create_layer(x_value, y_middle, network_shape, layer_height, layer_width, i)

  
  x_value += layer_width

x1 = 568
x2 = 568 + layer_width
x3 = 568 + layer_width * 2
x4 = 568 + layer_width * 3
svg_post = f"""
    <text class="text" dy=".35em" x="{x1}" y="610" style="font-size: 12px;">Input ∈ ℝ¹ x N</text>
    <text class="text" dy=".35em" x="{x2}" y="610" style="font-size: 12px;">Hidden ∈ ℝ⁵ x N</text>
    <text class="text" dy=".35em" x="{x3}" y="610" style="font-size: 12px;">Hidden ∈ ℝ⁵ x N</text>
    <text class="text" dy=".35em" x="{x4}" y="610" style="font-size: 12px;">Output ∈ ℝ² x N</text>
  </g>
  <defs>
    <marker id="arrow" viewBox="0 -5 10 10" markerWidth="7" markerHeight="7" orient="auto" refX="40">
      <path d="M0,-5L10,0L0,5" style="stroke: rgb(80, 80, 80); fill: none;"></path>
    </marker>
  </defs>
</svg>
"""
svg_text += svg_post

with open(f"svg_test.svg", "w") as f:
  f.write(svg_text)
