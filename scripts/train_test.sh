#train
python -m train.train_mdm --resume_checkpoint save/humanml_trans_enc_512/model000475000.pt --save_dir save/mdm_finetune_1122 --save_interval 5000 --lr 5e-5
python -m train.train_mdm --resume_checkpoint save/mdm_finetune/model000565000.pt --save_dir save/mdm_finetune_actions --save_interval 5000
python -m train.train_mdm --resume_checkpoint save/mdm_finetune_actions_reorder/model000600000.pt --save_dir save/mdm_finetune_actions_all_text_3 --save_interval 5000 --lr 5e-5
# sample
python -m sample.generate --model_path ./save/mdm_finetune/model000560000.pt --text_prompt "the person walked forward and is picking up his toolbox."
python -m sample.generate --model_path ./save/mdm_finetune_actions/model000600000.pt --text_prompt "the person walked forward and is picking up his toolbox."
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000475000.pt --text_prompt "the person walked forward and is picking up his toolbox."
python -m sample.generate --model_path ./save/mdm_finetune_actions_wo_physics/model000690000.pt --text_prompt "the person walked forward and is picking up his toolbox."
python -m sample.generate --model_path ./save/mdm_finetune_111920/model000515000.pt --text_prompt "the person walked forward and is picking up his toolbox."
# eval
python -m eval.eval_humanml --model_path ./save/mdm_finetune/model000565000.pt
python -m eval.eval_humanml --model_path ./save/mdm_finetune_actions/model000600000.pt
python -m eval.eval_humanml --model_path ./save/humanml_trans_enc_512/model000475000.pt
python -m eval.eval_humanml --model_path ./save/mdm_finetune_actions_wo_physics/model000690000.pt --eval_mode debug
python -m eval.eval_humanml --model_path ./save/mdm_finetune_actions_reorder/model000600000.pt
python -m eval.eval_humanml --model_path ./save/mdm_finetune_actions_all_text_4/model000635000.pt --eval_mode debug

python -m train.train_mdm --resume_checkpoint save/model000750000.pt --save_dir save/mdm_finetune_1119 --save_interval 5000
python -m train.train_mdm --resume_checkpoint save/mdm_finetune_1121/model000515000.pt --save_dir save/1120_wo_physics --save_interval 5000 --lr 5e-5
#visualize
python -m visualize.render_mesh --input_path ./save/mdm_finetune_initial_exp/mdm_finetune_actions_wo_physics/samples_mdm_finetune_actions_wo_physics_000690000_seed10_craw/sample00_rep00.mp4
