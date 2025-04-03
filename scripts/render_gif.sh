 # absolute path to this project
 project_dir="/home/jovyan/mogen/motion-diffusion-model"
 # absolute path to blender app
 blender_app="/home/jovyan/mogen/ProgMoGen/progmogen/blender-2.93.18-linux-x64/blender"
 save_dir="01_15_linear/all_text_1/samples_mdm_finetune_linear_all_texts_01_15"
 mesh_file="${project_dir}/save/${save_dir}"
 python3 -m visualize.render_mesh_each --input_path "${save_fig_dir}/gen.npy" --selected_idx ${idx}
 ${blender_app} --python render_demo_hoi1.py -- npy="gen-1_smpl_params.npy" +npy_joint="results.npy" +npy_joint_idx=-1 canonicalize=true mode="video"