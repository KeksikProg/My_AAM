import casadi as ca

def appearance_model(base_texture, blend_textures):
    base = ca.MX(base_texture)
    blends_mx = [ca.MX(t) - base for t in blend_textures]
    w_tex = ca.MX.sym("w_tex", len(blends_mx))
    
    appearance = base + ca.mtimes(ca.horzcat(*blends_mx), w_tex)
    return w_tex, appearance
