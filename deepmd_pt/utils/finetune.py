import logging

def get_model_type_map(origin_config, model_params):
    assert origin_config != {}, "Finetune model need to contain the config information."
    pretrained_type_map = origin_config["model"]["type_map"]
    cur_type_map = model_params["type_map"]
    out_line_type = []
    for i in cur_type_map:
        if i not in pretrained_type_map:
            out_line_type.append(i)
    assert not out_line_type, (
        "{} type(s) not contained in the pretrained model! "
        "Please choose another suitable one.".format(str(out_line_type))
    )
    if cur_type_map != pretrained_type_map:
        logging.info(
            "Change the type_map from {} to {}.".format(
                str(cur_type_map), str(pretrained_type_map)
            )
        )
    return pretrained_type_map, cur_type_map