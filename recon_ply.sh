
###
 # @Author: Easonyesheng preacher@sjtu.edu.cn
 # @Date: 2026-05-20 16:20:29
 # @LastEditors: Easonyesheng preacher@sjtu.edu.cn
 # @LastEditTime: 2026-05-20 16:24:48
 # @FilePath: /recon/ee_recon/third_party/ee-vggt-omega/recon_ply.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

echo "Reconstructing point cloud from images using VGG-T Omega..."
python reconstruct_ply.py \
  --checkpoint /opt/data/private/recon/weights/vggt-omega/vggt_omega_1b_512.pt \
  --input-dir /opt/data/private/recon/data/comac/global_20_subset \
  --output /opt/data/private/recon/ee_recon/third_party/ee-vggt-omega/outputs/comac_global.ply