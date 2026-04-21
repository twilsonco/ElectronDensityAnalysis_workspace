using DataFrames, PlutoUI
df = DataFrame(selected_x_col = [1.0, 5.0], filter_col = [2.0, 4.0])
selected_x = "selected_x_col"
filter_var = "filter_col"

x_min = floor(minimum(df[!, selected_x]) * 10) / 10
x_max = ceil(maximum(df[!, selected_x]) * 10) / 10
x_step = (x_max - x_min) / 100.0
x_step = x_step == 0 ? 0.1 : x_step
x_slider_obj = RangeSlider(x_min:x_step:x_max; default=x_min:x_step:x_max, show_value=true)
println("Success!")
