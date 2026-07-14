%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MMCLAB - Mesh-based CW adjoint Jacobian demo
%
% Demonstrates direct computation of the adjoint Jacobian on a tetrahedral
% mesh (PR#2 b1f77fd / ed4733e). Compared to the grid-mode adjoint (see
% existing mcxlab/example/demo_mcx_adjoint_jacobian.m), the mesh path:
%   - keeps the simulation on the native tet mesh (no voxelisation),
%   - returns J_mua and J_D as 2D arrays [mesh.nn, Ns*Nd],
%   - uses the FEM-integrated formula from rb_femjacobian (redbird-m
%     matlab/rbfemmatrix.cpp), which mirrors eq. 3d2d:adjoint:elemform
%     in Q. Fang PhD thesis (Thayer School of Engineering, Dartmouth
%     College), Chap. 6 sssec:3d3d:nodal.
%
% Three source-detector pairs are evaluated in a single Monte Carlo run.
% Output: J_mua and J_D banana profiles on the y=30 mid-plane.
%
% Requires: cfg.method = 'elem' (mesh mode) and cfg.basisorder = 1 (nodal
% fluence so del-phi is well-defined inside each tet).
%
% This file is part of Mesh-based Monte Carlo (MMC) URL:https://mcx.space/mmc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg;

% ---- Box mesh: 60 x 60 x 30 mm, refinement size 4 -------------------
[cfg.node, face, cfg.elem] = meshabox([0 0 0], [60 60 30], 4);
cfg.elemprop = ones(size(cfg.elem, 1), 1);

cfg.nphoton = 3e7;

% Single source at (30, 30, 0), pointing +z (into the slab)
cfg.srcpos  = [30 30 0];
cfg.srcdir  = [0  0  1];

% Three detectors at increasing s-d separation along +x.
% detpos column 4 = disk radius (mm); detdir gives the adjoint-source
% direction (back into the slab, here +z just like the forward source).
cfg.detpos = [20 30 0 1.5
              40 30 0 1.5
              45 30 0 1.5];
cfg.detdir = [0 0 1 0
              0 0 1 0
              0 0 1 0];

cfg.prop       = [0 0 1 1; 0.005 1 0 1.37];
cfg.tstart     = 0;
cfg.tend       = 5e-9;
cfg.tstep      = 5e-9;
cfg.method     = 'elem';                 % mesh mode (Branchless Badouel)
cfg.basisorder = 1;                      % nodal fluence required for J_D

% adjoint_mua_d -> dual output, fills both flux.jmua and flux.jd
cfg.outputtype  = 'adjoint_mua_d';
cfg.adjointmode = 0;                     % 0 = full FEM (default),
% 1 = nodal-adjoint approx
%     (rbjacmuafast.m formula)
cfg.debuglevel = 'TP';

fprintf('\n-- Running CW mesh-mode adjoint (Ns=1, Nd=3 -> 3 J_mua + 3 J_D) --\n');
flux = mmclab(cfg);

% flux.data : forward fluence per slot, shape [nn, maxgate, Ns+Nd] = [nn, 1, 4]
% flux.jmua : real single, shape [nn, Ns*Nd]   = [nn, 3]
% flux.jd   : real single, shape [nn, Ns*Nd]   = [nn, 3]
fprintf('  flux.data size: %s\n', mat2str(size(flux.data)));
fprintf('  flux.jmua size: %s   range: [%.2e, %.2e]\n', ...
        mat2str(size(flux.jmua)), min(flux.jmua(:)), max(flux.jmua(:)));
fprintf('  flux.jd   size: %s   range: [%.2e, %.2e]\n', ...
        mat2str(size(flux.jd)),   min(flux.jd(:)),   max(flux.jd(:)));

% ---- Plot the three banana profiles for J_mua and J_D ---------------
figure('Name', 'Mesh adjoint Jacobian (CW, mmclab)', ...
       'Position', [60 60 1300 720]);

npair = size(flux.jmua, 2);

for k = 1:npair
    subplot(2, npair, k);
    Jmua = double(flux.jmua(:, k));
    plotmesh([cfg.node, log10(abs(Jmua) + 1e-12)], cfg.elem, 'y=30', ...
             'facecolor', 'interp', 'linestyle', 'none');
    view([0 1 0]);
    colorbar;
    title(sprintf('|J_{\\mu_a}|  S-D_%d  (\\Deltax=%g mm)', ...
                  k, cfg.detpos(k, 1) - cfg.srcpos(1)));

    subplot(2, npair, npair + k);
    Jd = double(flux.jd(:, k));
    plotmesh([cfg.node, log10(abs(Jd) + 1e-12)], cfg.elem, 'y=30', ...
             'facecolor', 'interp', 'linestyle', 'none');
    view([0 1 0]);
    colorbar;
    title(sprintf('|J_D|  S-D_%d', k));
end
try
    sgtitle(['Mesh-mode adjoint (full FEM): J_{\mu_a} (top) and J_D (bottom)' ...
             '  for 3 S-D pairs']);
catch
    % sgtitle was added in Octave 8.0; older Octave silently skips it.
end

% ---- Sanity check: J_mua should be (essentially) phi_S * phi_D ------
% flux.data is [nn, 1, Ns+Nd] with slots 0..Ns-1 forward sources and
% Ns..Ns+Nd-1 detector-adjoint sources. Compare J_mua(S,D_1) against the
% direct nodal product phi_S * phi_D_1 (up to the FEM-vs-nodal-adjoint
% formula difference and the global -V_e/10 weighting).
phi_src = double(squeeze(flux.data(:, 1, 1)));        % slot 1 = forward S
phi_d1  = double(squeeze(flux.data(:, 1, 2)));        % slot 2 = first detector adjoint
fprintf('\nConsistency: corrcoef( J_mua(S,D_1) , phi_S * phi_D_1 ):\n');
prod_sd = phi_src .* phi_d1;
mask    = (abs(prod_sd) > 1e-12) & (abs(flux.jmua(:, 1)) > 1e-12);
cc      = corrcoef(log10(abs(prod_sd(mask))), log10(abs(double(flux.jmua(mask, 1)))));
fprintf('  log10-log10 corr coef = %.4f  (expected close to 1.0)\n', cc(1, 2));
