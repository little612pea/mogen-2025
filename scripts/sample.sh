#!/bin/bash
text="A person is fishing at the lake"
exp_name="01_15_linear"
model_name="wo_physics_then_all_text_2"
model_iter="000590000"
arch_decoupling="linear"
stage="full-text"
formatted_text_prompt=$(echo "$text" | sed 's/ /_/g' | sed 's/\./_/g')
out_path="samples_${model_name}_${model_iter}_seed10_${formatted_text_prompt}"
out_path_length=${#out_path}
if [ $out_path_length -gt 69 ]; then
    out_path="${out_path:0:69}"
fi
echo ${out_path}
# absolute path to this project
project_dir="/home/jovyan/mogen/motion-diffusion-model"
# absolute path to blender app
blender_app="/home/jovyan/mogen/ProgMoGen/progmogen/blender-2.93.18-linux-x64/blender"


cd ..
python -m sample.generate \
    --model_path ./save/${exp_name}/${model_name}/model${model_iter}.pt \
    --text_prompt "${text}"  \
    --arch_decoupling=${arch_decoupling} --stage=${stage}


save_dir="${exp_name}/${model_name}/${out_path}"
mesh_file="${project_dir}/save/${save_dir}"
cd ./scripts
python render_mesh_each.py --input_path "${mesh_file}/results.npy" --selected_idx -1
cp ${mesh_file}/results.npy ${project_dir}/TEMOS-master
cp ${mesh_file}/results_smpl/gen-1_smpl_params.npy ${project_dir}/TEMOS-master
cd ../TEMOS-master
${blender_app} --background --python render_demo_hoi1.py -- npy="gen-1_smpl_params.npy" +npy_joint="results.npy" +npy_joint_idx=-1 canonicalize=true mode="video"
cp ./gen-1_smpl_params.mp4 ${mesh_file}
rm gen-1_smpl_params.mp4

 