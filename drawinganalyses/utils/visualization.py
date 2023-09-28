import torch.nn.functional as F
import torch
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import numpy as np

from drawinganalyses.config import LOCAL_DATA_DIR


def explinability_images(dataloader, model, label_to_str, class_names, count_season):
    total_count = 0
    
    for inputs, labels in iter(dataloader):
        

        keep = True
        true_label = label_to_str[labels.item()]
        output = model(inputs)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        pred_label_idx.squeeze_()
        predicted_label = class_names[(pred_label_idx.item())]
        print('Predicted:', predicted_label, '(', prediction_score .squeeze().item(), ')')
        print('Ground truth:', true_label)
        print("total count :", total_count)
        
        if predicted_label == label_to_str[labels.item()] and count_season[predicted_label][0] < 5:
            count_season[predicted_label][0] += 1
            total_count += 1
        elif predicted_label != label_to_str[labels.item()] and count_season[predicted_label][1] < 5:
            count_season[predicted_label][1] += 1
            total_count += 1
        else:
            print("false")
            keep = False
        
        if keep:
            try:
                torch.manual_seed(0)
                np.random.seed(0)
                    
                occlusion = Occlusion(model)
                
                copy_inputs = inputs
                
                attributions_occ = occlusion.attribute(inputs,
                                    strides = (3, 8, 8),
                                    target=pred_label_idx,
                                    sliding_window_shapes=(3,15, 15),
                                    baselines=0)

                _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                    np.transpose(inputs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                    ["original_image", "heat_map", "heat_map", "masked_image"],
                                                    ["all", "positive", "negative", "positive"],
                                                    show_colorbar=True,
                                                    outlier_perc=2,
                                                    titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                                    fig_size=(18, 6)
                                                    )

                occlusion = Occlusion(model)

                attributions_occ1 = occlusion.attribute(copy_inputs,
                                            strides = (3, 50, 50),
                                            target=pred_label_idx,
                                            sliding_window_shapes=(3,60, 60),
                                            baselines=0)
                _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ1.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                    np.transpose(copy_inputs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                    ["original_image", "heat_map", "masked_image"],
                                                    ["all", "positive", "positive"],
                                                    show_colorbar=True,
                                                    outlier_perc=2,
                                                    titles=["Original", "Positive Attribution", "Masked"],
                                                    fig_size=(18, 6)
                                                )
            
                if total_count == 20:
                    break
                
            except(AssertionError):
                continue
        


def explinability_images_save(dataloader, model, label_to_str, class_names):
    total_count = 0
    
    for inputs, labels in iter(dataloader):
        
        true_label = label_to_str[labels.item()]
        output = model(inputs)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        pred_label_idx.squeeze_()
        predicted_label = class_names[(pred_label_idx.item())]
        print('Predicted:', predicted_label, '(', prediction_score .squeeze().item(), ')')
        print('Ground truth:', true_label)
        print("total count :", total_count)

    
        try:
            torch.manual_seed(0)
            np.random.seed(0)
                
            occlusion = Occlusion(model)
            
            copy_inputs = inputs
            
            attributions_occ = occlusion.attribute(inputs,
                                strides = (3, 8, 8),
                                target=pred_label_idx,
                                sliding_window_shapes=(3,15, 15),
                                baselines=0)

            plt_fig, plt_axis = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                ["original_image", "heat_map", "heat_map", "masked_image"],
                                                ["all", "positive", "negative", "positive"],
                                                show_colorbar=True,
                                                outlier_perc=2,
                                                titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                                fig_size=(18, 6)
                                                )

            occlusion = Occlusion(model)

            attributions_occ1 = occlusion.attribute(copy_inputs,
                                        strides = (3, 50, 50),
                                        target=pred_label_idx,
                                        sliding_window_shapes=(3,60, 60),
                                        baselines=0)
            
            plt_fig2, plt_axis2 = viz.visualize_image_attr_multiple(np.transpose(attributions_occ1.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                np.transpose(copy_inputs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                ["original_image", "heat_map", "masked_image"],
                                                ["all", "positive", "positive"],
                                                show_colorbar=True,
                                                outlier_perc=2,
                                                titles=["Original", "Positive Attribution", "Masked"],
                                                fig_size=(18, 6)
                                            )


            # Create a new figure (fig3) where you want to merge the two figures
            fig3 = plt.figure(figsize=(36,12))

            # Add subplots to fig3 where you want to place the contents of fig1 and fig2
            # In this example, we'll create a 1x2 grid of subplots
            ax3 = fig3.add_subplot(2, 1, 1)
            ax4 = fig3.add_subplot(2, 1, 2)

            # Copy the contents of fig1 and fig2 into the subplots of fig3
            ax3.imshow(plt_fig.canvas.renderer.buffer_rgba())
            ax4.imshow(plt_fig2.canvas.renderer.buffer_rgba())

            # You may need to adjust the subplot positions and sizes as per your requirement

            # Add titles to the subplots if needed
            ax3.set_title('True label : {true_label}, Predicted label : {predicted_label}'.format(true_label=true_label, predicted_label=predicted_label))

            # Show or save the merged figure
            fig3.savefig(LOCAL_DATA_DIR / "MollyExplicability" / "{total_count}.png".format(total_count=total_count))
            total_count += 1
            print('saved figure ', total_count)
                     
        except(AssertionError):
            print("continue")
            continue
        


