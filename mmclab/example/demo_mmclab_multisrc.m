%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MMCLAB - Multi-source forward simulation demo
%
% Demonstrates the post-PR#3 multi-source forward path:
%   - cfg.srcpos / cfg.srcdir can be an Mx{3,4} matrix; the first row
%     populates cfg.srcpos and the remaining rows populate the internal
%     cfg.srcdata[] list (mirrors mcxlab's convention).
%   - cfg.srcid = -1 simulates every slot in one run; the output
%     flux.data is laid out as [datalen, maxgate, nsrcslots] with one
%     fluence map per source slot.
%   - cfg.srcid = K (1-based, 1..M) launches only from row K so the
%     output collapses back to a single slot.
%
% This file is part of Mesh-based Monte Carlo (MMC) URL:https://mcx.space/mmc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg cfg1 cfg2 fluxall flux1;

% ---- Common geometry: 60 x 60 x 30 mm box mesh ----------------------
[node, face, elem] = meshabox([0 0 0], [60 60 30], 6);

% ---- Multi-source forward (srcid = -1, run every slot together) -----
cfg.nphoton    = 1e7;
cfg.node       = node;
cfg.elem       = elem;
cfg.elemprop   = ones(size(cfg.elem, 1), 1);

% Two pencil sources, each carrying 50% of the launch budget.
% First row -> cfg.srcpos; second row -> cfg.srcdata[0].
cfg.srcpos = [20 30 0  0.5
              40 30 0  0.5];
cfg.srcdir = [0  0  1  0
              0  0  1  0];
cfg.srcid  = -1;                         % simulate both slots, separate outputs

cfg.prop       = [0 0 1 1; 0.005 1 0 1.37];
cfg.tstart     = 0;
cfg.tend       = 5e-9;
cfg.tstep      = 5e-9;
cfg.method     = 'elem';                 % mesh mode (Branchless Badouel)
cfg.basisorder = 1;                      % nodal fluence
cfg.debuglevel = 'TP';

fprintf('\n-- Running multi-source forward (srcid=-1, 2 sources) --\n');
fluxall = mmclab(cfg);

% flux.data has shape [nn, maxgate, nsrcslots] in mesh mode with srcnum=1
fprintf('  flux.data size: %s   nsrcslots=%d\n', ...
        mat2str(size(fluxall.data)), size(fluxall.data, ndims(fluxall.data)));

% ---- Single-slot replay (srcid = 1) for cross-validation ------------
cfg1            = cfg;
cfg1.srcid      = 1;                     % only the first slot (matches srcpos row 1)
cfg1.nphoton    = cfg.nphoton / 2;       % half the budget -> matches the
% per-slot photon count of the
% srcid=-1 run above

fprintf('\n-- Running srcid=1 only (slot 1, %.0e photons) --\n', cfg1.nphoton);
flux1 = mmclab(cfg1);

% ---- Plot per-source fluence (y=30 plane) ---------------------------
slab = abs(squeeze(fluxall.data(:, 1, :)));     % [nn, 2]

figure('Name', 'Multi-source forward (mmclab)', 'Position', [60 60 1200 420]);
for k = 1:2
    subplot(1, 3, k);
    plotmesh([cfg.node, log10(slab(:, k) + 1e-12)], cfg.elem, 'y=30', ...
             'facecolor', 'interp', 'linestyle', 'none');
    view([0 1 0]);
    colorbar;
    title(sprintf('Slot %d  (S at %g,%g,%g, w=%g)', ...
                  k, cfg.srcpos(k, 1), cfg.srcpos(k, 2), ...
                  cfg.srcpos(k, 3), cfg.srcpos(k, 4)));
end

% Sanity check: srcid=1 alone should reproduce the srcid=-1 slot-1 map
subplot(1, 3, 3);
phi_srcid1 = abs(squeeze(flux1.data(:, 1, 1)));
plotmesh([cfg.node, log10(phi_srcid1 + 1e-12)], cfg.elem, 'y=30', ...
         'facecolor', 'interp', 'linestyle', 'none');
view([0 1 0]);
colorbar;
title('Single run: srcid=1 only');
% sgtitle was added in Octave 8.0; wrap so older Octave silently skips it.
try
    sgtitle('Multi-source forward via cfg.srcpos = [...; ...]');
catch
end

% Relative agreement between slot-1 of the multi-run and the dedicated
% single-source run (should be within Monte Carlo noise, a few percent).
mask = (slab(:, 1) > 1e-6);
relerr = abs(phi_srcid1(mask) - slab(mask, 1)) ./ slab(mask, 1);
fprintf('\nSlot-1 multi vs srcid=1 single-run agreement:\n');
fprintf('  median rel.err: %.2f %%   95th pct: %.2f %%\n', ...
        100 * median(relerr), 100 * prctile(relerr, 95));
