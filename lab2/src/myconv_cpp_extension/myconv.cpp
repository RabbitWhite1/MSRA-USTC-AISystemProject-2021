// MIT License

// Copyright (c) Microsoft Corporation.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE


#include <torch/extension.h>
#include <ATen/TensorIndexing.h>
#include <iostream>
#include <vector>

using namespace torch::indexing;

std::vector<torch::Tensor> myconv_forward(
    torch::Tensor input,
    torch::Tensor weight) 
{
    // in_channels, out_channels, kernel_size, kernel_size = weight.size()
    auto out_channels = weight.size(1);
    auto kernel_size = weight.size(2);
    // batch_size, in_channels, in_height, in_width = input.size()
    auto input_sizes = input.sizes();
    auto out_height = input_sizes[2] - kernel_size + 1;
    auto out_width = input_sizes[3] - kernel_size + 1;
    auto out = torch::zeros({input_sizes[0], out_channels, out_height, out_width}).cuda();
    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            // std::cout << out.index({Slice(), Slice(), i, j}).sizes() << std::endl;
            // std::cout << input.index({Slice(), Slice(), None, Slice(i, i+kernel_size), Slice(j, j+kernel_size)}).sizes() << std::endl;
            // std::cout << weight.index({None}).sizes() << std::endl;
            out.index({Slice(), Slice(), i, j}) = 
                ((input.index({Slice(), Slice(), None, Slice(i, i+kernel_size), Slice(j, j+kernel_size)})) * weight.index({None, Slice()})).sum({1, 3, 4});
        }
    }

    return {out};
}

std::vector<torch::Tensor> myconv_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight
    ) 
{
    auto kernel_size = weight.size(2);
    auto grad_output_sizes = grad_output.sizes();
    auto out_height = grad_output_sizes[2];
    auto out_width = grad_output_sizes[3];
    // initialize grad
    auto grad_input = torch::zeros_like(input);
    auto grad_weight = torch::zeros_like(weight);
    
    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            grad_input.index({Slice(), Slice(), Slice(i, i+kernel_size), Slice(j, j+kernel_size)}) +=
                (grad_output.index({Slice(), None, Slice(), i, j, None, None}) * weight.index({None, Slice()})).sum(2);
        }
    }
    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            grad_weight += (
                    grad_output.index({Slice(), None, Slice(), i, j, None, None}) 
                        * input.index({Slice(), Slice(), None, Slice(i, i+kernel_size), Slice(j, j+kernel_size)})
                ).sum(0);
        }
    }
                
    return {grad_input, grad_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &myconv_forward, "myConv forward");
    m.def("backward", &myconv_backward, "myConv backward");
}
