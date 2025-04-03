python render_mesh_each.py --input_path "results.npy" --selected_idx -1
cp results.npy ../TEMOS-master
cp results_smpl/gen-1_smpl_params.npy ../TEMOS-master
cd ../TEMOS-master
blender_app="/home/jovyan/mogen/ProgMoGen/progmogen/blender-2.93.18-linux-x64/blender"
${blender_app} --background --python render_demo_hoi1.py -- npy="gen-1_smpl_params.npy" +npy_joint="results.npy" +npy_joint_idx=-1 canonicalize=true mode="video"

 