import numpy as np
import torch
import torch.nn as nn



def prune_by_percentile_magnitude_perLayer(percent_pruning_min, percent_pruning_max, model, mask):


    statistic = {}
    new_masks = {}
    for name, param in model.named_parameters():
        if "prompt" in name or 'head' in name:
            old_mask = mask[name]
            new_mask = old_mask - 1
        elif "adapter" in name:
            layer_ind = name.split("blocks.")[-1].split(".")[0]
            layer_name = f"blocks.{layer_ind}"
            module_name = name.split(layer_name + ".")[1]
            old_mask = mask[layer_name][module_name]
            new_mask = old_mask - 1
        else:
            # Calculate percentile value
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values

            percentile_value_min = np.percentile(abs(alive), percent_pruning_min)
            percentile_value_max = np.percentile(abs(alive), percent_pruning_max)

            if "blocks" in name:
                layer_ind = name.split("blocks.")[-1].split(".")[0]
                layer_name = f"blocks.{layer_ind}"
                module_name = name.split(layer_name+".")[1]
                old_mask = mask[layer_name][module_name]
            else:
                old_mask = mask[name]

            tensor = param.data.cpu().numpy()
            new_mask = np.where((abs(tensor) > percentile_value_min) & (abs(tensor) < percentile_value_max), 0, old_mask)


        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)")

        if "blocks" in name:
            if layer_name not in new_masks:
                new_masks[layer_name] = {}
            new_masks[layer_name][module_name] = torch.from_numpy(new_mask).cuda()
        else:
            new_masks[name] = torch.from_numpy(new_mask).cuda()

    statistic_ = {}
    for na, [trainable_p, t_p] in statistic.items():
        if 'blocks' in na:
            layer_ind = na.split('blocks.')[-1].split(".")[0]
            layer_name = 'blocks.' + layer_ind
            if layer_name not in statistic_:
                statistic_[layer_name] = [trainable_p, t_p]
            else:
                statistic_[layer_name] = [statistic_[layer_name][0] + trainable_p, statistic_[layer_name][1] + t_p]
        else:
            statistic_[na] = [trainable_p, t_p]
    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic_.items():
        print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

def prune_by_random_perLayer(percent_pruning_min, percent_pruning_max, model, mask):

    statistic = {}
    new_masks = {}
    for name, param in model.named_parameters():
        if "prompt" in name or 'head' in name:
            old_mask = mask[name]
            new_mask = old_mask - 1
        elif "adapter" in name:
            layer_ind = name.split("blocks.")[-1].split(".")[0]
            layer_name = f"blocks.{layer_ind}"
            module_name = name.split(layer_name + ".")[1]
            old_mask = mask[layer_name][module_name]
            new_mask = old_mask - 1
        else:
            # Calculate percentile value
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values

            percentile_value_min = np.percentile(abs(alive), percent_pruning_min)
            percentile_value_max = np.percentile(abs(alive), percent_pruning_max)

            if "blocks" in name:
                layer_ind = name.split("blocks.")[-1].split(".")[0]
                layer_name = f"blocks.{layer_ind}"
                module_name = name.split(layer_name+".")[1]
                old_mask = mask[layer_name][module_name]
            else:
                old_mask = mask[name]

            tensor = param.data.cpu().numpy()
            new_mask = np.where((abs(tensor) > percentile_value_min) & (abs(tensor) < percentile_value_max), 0, old_mask)
            np.random.shuffle(new_mask)

        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)")

        if "blocks" in name:
            if layer_name not in new_masks:
                new_masks[layer_name] = {}
            new_masks[layer_name][module_name] = torch.from_numpy(new_mask).cuda()
        else:
            new_masks[name] = torch.from_numpy(new_mask).cuda()

    statistic_ = {}
    for na, [trainable_p, t_p] in statistic.items():
        if 'blocks' in na:
            layer_ind = na.split('blocks.')[-1].split(".")[0]
            layer_name = 'blocks.' + layer_ind
            if layer_name not in statistic_:
                statistic_[layer_name] = [trainable_p, t_p]
            else:
                statistic_[layer_name] = [statistic_[layer_name][0] + trainable_p, statistic_[layer_name][1] + t_p]
        else:
            statistic_[na] = [trainable_p, t_p]
    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic_.items():
        print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

def prune_by_percentile_gradient_perLayer(percent_pruning_min, percent_pruning_max, model, mask):

    statistic = {}
    new_masks = {}
    for name, param in model.named_parameters():
        if "prompt" in name or 'head' in name:
            old_mask = mask[name]
            new_mask = old_mask - 1
        elif "adapter" in name:
            layer_ind = name.split("blocks.")[-1].split(".")[0]
            layer_name = f"blocks.{layer_ind}"
            module_name = name.split(layer_name + ".")[1]
            old_mask = mask[layer_name][module_name]
            new_mask = old_mask - 1
        elif "ssf" in name:
            if "blocks" in name:
                layer_ind = name.split("blocks.")[-1].split(".")[0]
                layer_name = f"blocks.{layer_ind}"
                module_name = name.split(layer_name + ".")[1]
                old_mask = mask[layer_name][module_name]
                new_mask = old_mask - 1
            else:
                old_mask = mask[name]
                new_mask = old_mask - 1
        else:
            # Calculate percentile value
            tensor = param.grad.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values

            # if "cls_token" in name:
            #     percentile_value_min = np.percentile(abs(alive), 0)
            #     percentile_value_max = np.percentile(abs(alive), 100)
            if len(alive) <1000:
                percentile_value_min = np.percentile(abs(alive), 100)
                percentile_value_max = np.percentile(abs(alive), 100)
            else:
                percentile_value_min = np.percentile(abs(alive), percent_pruning_min)
                percentile_value_max = np.percentile(abs(alive), percent_pruning_max)
            # percentile_value_min = np.percentile(abs(alive), percent_pruning_min)
            # percentile_value_max = np.percentile(abs(alive), percent_pruning_max)

            if "blocks" in name:
                layer_ind = name.split("blocks.")[-1].split(".")[0]
                layer_name = f"blocks.{layer_ind}"
                module_name = name.split(layer_name+".")[1]
                old_mask = mask[layer_name][module_name]
            else:
                old_mask = mask[name]

            tensor = param.grad.data.cpu().numpy()
            new_mask = np.where((abs(tensor) > percentile_value_min) & (abs(tensor) <= percentile_value_max), 0, old_mask)


        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape)

        if "blocks" in name:
            if layer_name not in new_masks:
                new_masks[layer_name] = {}
            new_masks[layer_name][module_name] = torch.from_numpy(new_mask).cuda()
        else:
            new_masks[name] = torch.from_numpy(new_mask).cuda()

    statistic_ = {}
    for na, [trainable_p, t_p] in statistic.items():
        if 'blocks' in na:
            layer_ind = na.split('blocks.')[-1].split(".")[0]
            layer_name = 'blocks.' + layer_ind
            if layer_name not in statistic_:
                statistic_[layer_name] = [trainable_p, t_p]
            else:
                statistic_[layer_name] = [statistic_[layer_name][0] + trainable_p, statistic_[layer_name][1] + t_p]
        else:
            statistic_[na] = [trainable_p, t_p]
    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic_.items():
        print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

def prune_by_percentile_gradient_perCell(model, mask, time_para=1):
    statistic = {}
    new_masks = {}
    for name, param in model.named_parameters():

        if "norm" in name or "pos_embed" in name or "cls_token" in name:
            if "blocks" in name:
                layer_ind = name.split("blocks.")[-1].split(".")[0]
                layer_name = f"blocks.{layer_ind}"
                module_name = name.split(layer_name + ".")[1]
                new_mask = mask[layer_name][module_name]
            else:
                new_mask = mask[name]
        elif 'head' in name or "bias" in name:
            if "blocks" in name:
                layer_ind = name.split("blocks.")[-1].split(".")[0]
                layer_name = f"blocks.{layer_ind}"
                module_name = name.split(layer_name + ".")[1]
                old_mask = mask[layer_name][module_name]
            else:
                old_mask = mask[name]
            new_mask = old_mask - 1
        else:
            if "patch_embed" in name:
                tensor = param.grad.data.cpu().numpy()
                B,C,H,W = tensor.shape
                tensor = np.reshape(tensor,[B,-1])
            else:
                tensor = param.grad.data.cpu().numpy()

            new_mask=np.ones_like(tensor)
            for ind in range(time_para):
                max_index = abs(tensor).argsort(1)[:, -(ind + 1)]
                one_hot_temp = ~np.eye(max(tensor.shape))[max_index][:, :tensor.shape[1]].astype(np.bool)
                new_mask_temp = one_hot_temp.astype(np.float32)
                new_mask = new_mask.astype(int) & new_mask_temp.astype(int)
                new_mask = new_mask.astype(np.float32)
                print(f"88888888888888888888888 time_para: {time_para} 888888888888888888888888888")


            if "patch" in name:
                new_mask = np.reshape(new_mask, (B,C,H,W))

        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   )

        if "blocks" in name:
            layer_ind = name.split("blocks.")[-1].split(".")[0]
            layer_name = f"blocks.{layer_ind}"
            module_name = name.split(layer_name + ".")[1]

            if layer_name not in new_masks:
                new_masks[layer_name] = {}
            new_masks[layer_name][module_name] = torch.from_numpy(new_mask).cuda()
        else:
            new_masks[name] = torch.from_numpy(new_mask).cuda()

    statistic_ = {}
    for na, [trainable_p, t_p] in statistic.items():
        if 'blocks' in na:
            layer_ind = na.split('blocks.')[-1].split(".")[0]
            layer_name = 'blocks.' + layer_ind
            if layer_name not in statistic_:
                statistic_[layer_name] = [trainable_p, t_p]
            else:
                statistic_[layer_name] = [statistic_[layer_name][0] + trainable_p, statistic_[layer_name][1] + t_p]
        else:
            statistic_[na] = [trainable_p, t_p]
    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic_.items():
        print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

def prune_by_percentile_gradient_perCellAndperLayer(percent_pruning_min, percent_pruning_max, model, mask):
    statistic = {}
    new_masks = {}
    for name, param in model.named_parameters():
        if "norm" in name or "position_embeddings" in name or "cls_token" in name:
            if "blocks" in name:
                layer_ind = name.split("blocks.")[-1].split(".")[0]
                layer_name = f"blocks.{layer_ind}"
                module_name = name.split(layer_name + ".")[1]
                new_mask = mask[layer_name][module_name]
            else:
                new_mask = mask[name]
        elif 'head' in name or "bias" in name:
        # elif 'head' in name or "bias" in name or "norm" in name:
        # elif 'head' in name:
            if "blocks" in name:
                layer_ind = name.split("blocks.")[-1].split(".")[0]
                layer_name = f"blocks.{layer_ind}"
                module_name = name.split(layer_name + ".")[1]
                old_mask = mask[layer_name][module_name]
            else:
                old_mask = mask[name]
            new_mask = old_mask - 1
        else:
            if "patch_embeddings" in name:
                tensor = param.grad.data.cpu().numpy()
                B,C,H,W = tensor.shape
                tensor = np.reshape(tensor,[B,-1])
            else:
                tensor = param.grad.data.cpu().numpy()

            max_index = abs(tensor).argsort(0)[-1, :]
            one_hot = ~np.eye(max(tensor.shape))[max_index].T[:tensor.shape[0], :].astype(np.bool)
            new_mask = one_hot.astype(np.float32)

            if "patch" in name:
                new_mask = np.reshape(new_mask, (B,C,H,W))

        #pruning per layer
        tensor = param.grad.data.cpu().numpy()
        alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
        if len(alive) <1000:
            percentile_value_min = np.percentile(abs(alive), 100)
            percentile_value_max = np.percentile(abs(alive), 100)
        else:
            percentile_value_min = np.percentile(abs(alive), percent_pruning_min)
            percentile_value_max = np.percentile(abs(alive), percent_pruning_max)
        tensor = param.grad.data.cpu().numpy()
        new_mask = np.where((abs(tensor) > percentile_value_min) & (abs(tensor) <= percentile_value_max), 0, new_mask)


        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)")

        if "blocks" in name:
            layer_ind = name.split("blocks.")[-1].split(".")[0]
            layer_name = f"blocks.{layer_ind}"
            module_name = name.split(layer_name + ".")[1]

            if layer_name not in new_masks:
                new_masks[layer_name] = {}
            new_masks[layer_name][module_name] = torch.from_numpy(new_mask).cuda()
        else:
            new_masks[name] = torch.from_numpy(new_mask).cuda()

    statistic_ = {}
    for na, [trainable_p, t_p] in statistic.items():
        if 'blocks' in na:
            layer_ind = na.split('blocks.')[-1].split(".")[0]
            layer_name = 'blocks.' + layer_ind
            if layer_name not in statistic_:
                statistic_[layer_name] = [trainable_p, t_p]
            else:
                statistic_[layer_name] = [statistic_[layer_name][0] + trainable_p, statistic_[layer_name][1] + t_p]
        else:
            statistic_[na] = [trainable_p, t_p]
    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic_.items():
        print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

def prune_by_percentile_gradient_perCellAndallLayer(percent_pruning_min, percent_pruning_max, model, mask):

    # Calculate percentile value
    alive_all = np.array([])
    for name, param in model.named_parameters():
        if "head" in name: continue
        tensor = param.grad.data.cpu().numpy()
        alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
        alive_all = np.concatenate([alive_all, alive])

    percentile_value_min = np.percentile(abs(alive_all), percent_pruning_min)
    percentile_value_max = np.percentile(abs(alive_all), percent_pruning_max)


    statistic = {}
    new_masks = {}
    for name, param in model.named_parameters():
        if "norm" in name or "position_embeddings" in name or "cls_token" in name:
            if "blocks" in name:
                layer_ind = name.split("blocks.")[-1].split(".")[0]
                layer_name = f"blocks.{layer_ind}"
                module_name = name.split(layer_name + ".")[1]
                new_mask = mask[layer_name][module_name]
            else:
                new_mask = mask[name]
        elif 'head' in name or "bias" in name:
        # elif 'head' in name or "bias" in name or "norm" in name:
        # elif 'head' in name:
            if "blocks" in name:
                layer_ind = name.split("blocks.")[-1].split(".")[0]
                layer_name = f"blocks.{layer_ind}"
                module_name = name.split(layer_name + ".")[1]
                old_mask = mask[layer_name][module_name]
            else:
                old_mask = mask[name]
            new_mask = old_mask - 1
        else:
            if "patch_embeddings" in name:
                tensor = param.grad.data.cpu().numpy()
                B,C,H,W = tensor.shape
                tensor = np.reshape(tensor,[B,-1])
            else:
                tensor = param.grad.data.cpu().numpy()

            max_index = tensor.argsort(0)[-1, :]
            one_hot = ~np.eye(max(tensor.shape))[max_index].T[:tensor.shape[0], :].astype(np.bool)
            new_mask = one_hot.astype(np.float32)

            if "patch" in name:
                new_mask = np.reshape(new_mask, (B,C,H,W))

        #pruning per layer
        tensor = param.grad.data.cpu().numpy()
        new_mask = np.where((abs(tensor) > percentile_value_min) & (abs(tensor) <= percentile_value_max), 0, new_mask)


        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)")

        if "blocks" in name:
            layer_ind = name.split("blocks.")[-1].split(".")[0]
            layer_name = f"blocks.{layer_ind}"
            module_name = name.split(layer_name + ".")[1]

            if layer_name not in new_masks:
                new_masks[layer_name] = {}
            new_masks[layer_name][module_name] = torch.from_numpy(new_mask).cuda()
        else:
            new_masks[name] = torch.from_numpy(new_mask).cuda()

    statistic_ = {}
    for na, [trainable_p, t_p] in statistic.items():
        if 'blocks' in na:
            layer_ind = na.split('blocks.')[-1].split(".")[0]
            layer_name = 'blocks.' + layer_ind
            if layer_name not in statistic_:
                statistic_[layer_name] = [trainable_p, t_p]
            else:
                statistic_[layer_name] = [statistic_[layer_name][0] + trainable_p, statistic_[layer_name][1] + t_p]
        else:
            statistic_[na] = [trainable_p, t_p]
    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic_.items():
        print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

def prune_by_percentile_gradient_allLayer(percent_pruning_min, percent_pruning_max, model, mask):
    # Calculate percentile value
    alive_all = np.array([])
    for name, param in model.named_parameters():
        if "head" in name: continue
        tensor = param.grad.data.cpu().numpy()
        alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
        alive_all = np.concatenate([alive_all, alive])

    percentile_value_min = np.percentile(abs(alive_all), percent_pruning_min)
    percentile_value_max = np.percentile(abs(alive_all), percent_pruning_max)


    statistic = {}
    new_masks = {}
    for name, param in model.named_parameters():

        if "blocks" in name:
            layer_ind = name.split("blocks.")[-1].split(".")[0]
            layer_name = f"blocks.{layer_ind}"
            module_name = name.split(layer_name+".")[1]
            old_mask = mask[layer_name][module_name]
        else:
            old_mask = mask[name]

        tensor = param.grad.data.cpu().numpy()
        if "head" in name:
            new_mask = old_mask-1
        else:
            new_mask = np.where((abs(tensor) > percentile_value_min) & (abs(tensor) < percentile_value_max), 0, old_mask)
        trainable_param = (len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0]))
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)")

        if "blocks" in name:
            layer_ind = name.split("blocks.")[-1].split(".")[0]
            layer_name = f"blocks.{layer_ind}"
            module_name = name.split(layer_name + ".")[1]
            if layer_name not in new_masks:
                new_masks[layer_name] = {}
            new_masks[layer_name][module_name] = torch.from_numpy(new_mask).cuda()
        else:
            new_masks[name] = torch.from_numpy(new_mask).cuda()

    statistic_ = {}
    for na, [trainable_p, t_p] in statistic.items():
        if 'blocks' in na:
            layer_ind = na.split('blocks.')[-1].split(".")[0]
            layer_name = 'blocks.' + layer_ind
            if layer_name not in statistic_:
                statistic_[layer_name] = [trainable_p, t_p]
            else:
                statistic_[layer_name] = [statistic_[layer_name][0] + trainable_p, statistic_[layer_name][1] + t_p]
        else:
            statistic_[na] = [trainable_p, t_p]
    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic_.items():
        print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

def make_mask_(model):
    masks={}
    for name, param in model.named_parameters():
        if "blocks" in name:
            layer_ind = name.split("blocks.")[-1].split(".")[0]
            layer_name = f"blocks.{layer_ind}"
            module_name = name.split(layer_name+".")[1]
            if layer_name not in masks:
                masks[layer_name]={}
            masks[layer_name][module_name] = np.ones_like(param.data.cpu().numpy())
        else:
            masks[name] = np.ones_like(param.data.cpu().numpy())
    return masks
