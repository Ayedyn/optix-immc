%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MMCLAB - Mesh-based RF (frequency-domain) adjoint Jacobian demo
%
% Same setup as demo_mmclab_mesh_adjoint.m, but with cfg.omega > 0 to
% switch the forward solver into RF mode. The mesh BLBadouel kernel
% now tracks complex weight (r->weight_re + i*r->weight_im) per
% photon step using the same complex Beer-Lambert formula already
% used in the rtBLBadouelGrid (DMMC) path:
%
%     deposit(r, omega) = (w0 - w_new) / (mua + i*omega*n/c0)
%
% with w_new = w0 * exp(-mua*L) * exp(-i*omega*n*L/c0) per step.
% The kernel writes Re/Im fluence into separate buffer slots; the
% basisorder=1 mesh reduction now scatters both onto nodes; the host
% adjoint post-processing combines phi_s and phi_d into a complex
% J_mua per source-detector pair following the same Re/Im pattern
% as rb_femjacobian (redbird-m matlab/rbfemmatrix.cpp):
%
%     Re(J_mua) = Re(phi_s)*Re(phi_d) - Im(phi_s)*Im(phi_d)
%     Im(J_mua) = Re(phi_s)*Im(phi_d) + Im(phi_s)*Re(phi_d)
%
% Output: flux.jmua is complex single, shape [nn, Ns*Nd].
%
% This file is part of Mesh-based Monte Carlo (MMC) URL:https://mcx.space/mmc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg;

% ---- Box mesh: 60 x 60 x 30 mm -------------------------------------
[cfg.node, face, cfg.elem] = meshabox([0 0 0], [60 60 30], 4);
cfg.elemprop = ones(size(cfg.elem, 1), 1);

cfg.nphoton = 3e7;
cfg.srcpos  = [30 30 0];
cfg.srcdir  = [0  0  1];

% Two detectors at +/-10 mm from the source
cfg.detpos = [20 30 0 1.5
              40 30 0 1.5];
cfg.detdir = [0 0 1 0
              0 0 1 0];

cfg.prop       = [0 0 1 1; 0.005 1 0 1.37];
cfg.tstart     = 0;
cfg.tend       = 5e-9;
cfg.tstep      = 5e-9;
cfg.method     = 'elem';
cfg.basisorder = 1;                      % nodal fluence required for adjoint

% RF: 100 MHz modulation. cfg.omega is in rad/s, so omega = 2*pi*f_Hz.
cfg.omega = 2 * pi * 100e6;

cfg.outputtype  = 'adjoint';             % flux.jmua only (complex)
cfg.adjointmode = 0;                     % 0 = full FEM
cfg.debuglevel  = 'TP';

fprintf('\n-- Running RF mesh-mode adjoint at %.0f MHz --\n', ...
        cfg.omega / (2 * pi * 1e6));
flux = mmclab(cfg);

% flux.jmua is complex single, shape [nn, Ns*Nd] = [nn, 2]
% flux.data is complex double, shape [nn, maxgate, Ns+Nd]
fprintf('  flux.jmua complex: %d   size: %s\n', ...
        ~isreal(flux.jmua), mat2str(size(flux.jmua)));
fprintf('  |jmua| range: [%.2e, %.2e]\n', ...
        min(abs(flux.jmua(:))), max(abs(flux.jmua(:))));
fprintf('  flux.data complex: %d   size: %s\n', ...
        ~isreal(flux.data), mat2str(size(flux.data)));

% ---- Plot amplitude and phase for each S-D pair --------------------
figure('Name', 'Mesh RF adjoint Jacobian (mmclab)', ...
       'Position', [60 60 1200 720]);

npair = size(flux.jmua, 2);

for k = 1:npair
    Jamp = abs(double(flux.jmua(:, k)));
    Jphi = angle(double(flux.jmua(:, k))) * 180 / pi;

    subplot(2, npair, k);
    plotmesh([cfg.node, log10(Jamp + 1e-12)], cfg.elem, 'y=30', ...
             'facecolor', 'interp', 'linestyle', 'none');
    view([0 1 0]);
    colorbar;
    title(sprintf('|J_{\\mu_a}|  S-D_%d  (RF 100 MHz)', k));

    subplot(2, npair, npair + k);
    plotmesh([cfg.node, Jphi], cfg.elem, 'y=30', ...
             'facecolor', 'interp', 'linestyle', 'none');
    view([0 1 0]);
    colorbar;
    title(sprintf('arg(J_{\\mu_a})  S-D_%d  (deg)', k));
end
try
    sgtitle('Mesh-mode RF (100 MHz) adjoint Jacobian J_{\mu_a}: amplitude and phase');
catch
end

% ---- Consistency: J_mua ?= phi_S * phi_D (complex product) ----------
phi_src = double(squeeze(flux.data(:, 1, 1)));    % complex if RF
phi_d1  = double(squeeze(flux.data(:, 1, 2)));
prod_sd = phi_src .* phi_d1;
mask    = (abs(prod_sd) > 1e-12) & (abs(flux.jmua(:, 1)) > 1e-12);

if any(mask)
    cc_amp = corrcoef(log10(abs(prod_sd(mask))), log10(abs(double(flux.jmua(mask, 1)))));
    fprintf('\nConsistency (log|.| corrcoef):  |J_mua(S,D_1)| vs |phi_S * phi_D_1| = %.4f\n', ...
            cc_amp(1, 2));
    % phase agreement: J_mua's complex phase should track phi_S * phi_D_1's
    % phase up to a global FEM weighting that's real (-V_e/10) and therefore
    % adds no phase offset.
    dphase = mod(angle(double(flux.jmua(mask, 1))) - angle(prod_sd(mask)) + pi, 2 * pi) - pi;
    fprintf('Phase difference (rad):  median %.4f   max |.| %.4f\n', ...
            median(dphase), max(abs(dphase)));
end
