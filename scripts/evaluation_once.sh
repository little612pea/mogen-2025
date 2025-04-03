python -m eval.eval_humanml \
        --model_path ./save/DiP_no-target_10steps_context20_predict40/model000200000.pt \
        --eval_mode debug \
        --arch_decoupling="none" \
        --stage="full-text" \
        --use_ema --arch trans_dec --text_encoder_type bert 

python -m eval.eval_humanml \
        --model_path ./save/humanml_trans_enc_512/model000475000.pt \
        --eval_mode debug \
        --arch_decoupling="none" \
        --stage="full-text" 

python -m eval.eval_humanml \
        --model_path save/DiP_no-target_10steps_context20_predict40/model000600343.pt  --autoregressive \
        --use_ema --guidance_param 7.5 --stage "full-text" --arch_decoupling="none" --eval_encoder "agnostic"