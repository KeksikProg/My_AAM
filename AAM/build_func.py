import numpy as np
from warp_utils import warp_piecewise_affine

def build_func(mean_texture, appearance_deltas, target_image, TEXTURE_SIZE,
               base_shape, shape_deltas, triangles):

    num_app_params = appearance_deltas.shape[0]
    num_shape_params = shape_deltas.shape[0] + 2  # shape_deltas.shape == (N, 68, 2)

    def compute_warped_image(shape_params_val):
        # shape_params_val: (N,)
        tx, ty = shape_params_val[-2], shape_params_val[-1]
        shape_offset = np.tensordot(shape_params_val[:-2], shape_deltas, axes=(0, 0))  # (68, 2)
        shape = base_shape + shape_offset + np.array([tx, ty])
        warped = warp_piecewise_affine(target_image, shape, base_shape, triangles, TEXTURE_SIZE)
        return warped.flatten() / 255.0

    def builder(current_params):
        app_p = current_params[:num_app_params]          # (A,)
        shape_p = current_params[num_app_params:]        # (S,)

        eps = 1e-2
        base_warp = compute_warped_image(shape_p)
        

        # finite difference Jacobian for shape
        warped_plus = []
        warped_minus = []

        for i in range(num_shape_params):
            delta = np.zeros_like(shape_p)
            delta[i] = eps
            wp = compute_warped_image(shape_p + delta)
            wm = compute_warped_image(shape_p - delta)
            warped_plus.append(wp)
            warped_minus.append(wm)
            #print(f"Δ[{i}] = {np.linalg.norm(wp - wm):.6f}")

        J_shape = np.stack([(p - m) / (2 * eps) for p, m in zip(warped_plus, warped_minus)], axis=1)

        # model texture from appearance
        model_texture = mean_texture + appearance_deltas.T @ app_p  # (N,)
        residual = model_texture - base_warp  # (N,)

        J_app = appearance_deltas.T  # (N, A)
        J_full = np.hstack([J_app, -J_shape])  # (N, A+S)


        JTJ = J_full.T @ J_full
        JTr = J_full.T @ residual
        loss = np.mean(residual ** 2)

        return {
            "JTJ": JTJ,
            "JTr": JTr,
            "loss": loss
        }

    return builder
