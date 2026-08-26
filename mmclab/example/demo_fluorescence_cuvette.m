%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  two-pass excitation/emission ICG cuvette fluorescence demo
%
%  Reproduces the simplest validated case from the MCX-ExEm two-pass
%  excitation/emission method (Quel Imaging), matching:
%  Nguyen et al., "MCX-ExEm: a GPU Monte Carlo tool for simulating
%  fluorescence excitation and emission", J. Biomed. Opt. (2025).
%
%  Geometry/optics below are taken directly from that paper's own ICG
%  cuvette case: 10x10x20mm box, mua_background=0.002/mm, mus=0, g=0,
%  n=1.33, ICG epsilon_ex=17500 / epsilon_em=8500 M^-1mm^-1 @ 785/820nm,
%  QY=1, concentration=300nM. This demo uses cfg.srctype='elembary'
%  (mmclab.m) for Pass 2, the same mesh-native volumetric source used by
%  demo_mcxyz_skinvessel_shallowvessel.m's vessel fluorescence estimate.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg cfg2 flux fluxem;

%% ICG optical properties at 300nM

conc = 300e-9;     % 300 nM, in mol/L
epsilonEx = 17500; % M^-1 mm^-1 @ 785nm
epsilonEm = 8500;  % M^-1 mm^-1 @ 820nm
QY = 1;
muaBackground = 0.002; % 1/mm
muaIcgEx = 2.303 * epsilonEx * conc;
muaIcgEm = 2.303 * epsilonEm * conc;
muaTotalEx = muaBackground + muaIcgEx;
pFluor = (muaIcgEx / muaTotalEx) * QY;

%% build a simple regular tetrahedral mesh for the 10x10x20mm cuvette

step = 0.5; % mm
[cfg.node, cfg.elem] = genT6mesh(0:step:10, 0:step:10, 0:step:20);
cfg.elemprop = ones(size(cfg.elem, 1), 1);
cfg.unitinmm = 1;
cfg.method = 'elem';

%% Pass 1: excitation (785nm), pencil beam through the cuvette center
% (the paper's 5mm disk needs a widefield mesh extension tetgen fails to
% boundary-recover on this grid; a pencil beam is collimated the same way since mus=0)

cfg.prop = [0 0 1 1          % 0: background/ambient
            muaTotalEx 0 0 1.33]; % 1: ICG cuvette medium at 785nm

cfg.srcpos = [5 5 0];
cfg.srcdir = [0 0 1];
cfg.srctype = 'pencil';

[cfg.elem, cfg.evol] = meshreorient(cfg.node, cfg.elem(:, 1:4));
cfg.isreoriented = 1;

cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = 5e-9;
cfg.outputtype = 'energy';
cfg.basisorder = 0;
cfg.nphoton = 1e6;
cfg.seed = 1648335518;
cfg.debuglevel = 'TP';
cfg.isreflect = 0;

flux = mmclab(cfg);
absorbedfrac = flux.data;

%% Pass 2: emission (820nm) via a weighted-element + barycentric volumetric source

w = absorbedfrac(cfg.elemprop == 1) * pFluor;
cuvetteidx = find(cfg.elemprop == 1);
mask = w > 0;
srcidx = cuvetteidx(mask);
w = w(mask);
fprintf(1, 'fluorescence source: %d of %d cuvette elements have nonzero weight\n', ...
        numel(srcidx), numel(cuvetteidx));
fprintf(1, 'total emission weight budget (fraction of Pass 1 launched weight): %g\n', sum(w));

cfg2 = cfg;
cfg2.evol = abs(cfg.evol);
cfg2.srctype = 'elembary';
cfg2.srcparam1 = [numel(srcidx) 0 0 0];
cfg2.srcpattern = reshape([cumsum(w)'; srcidx'], [], 1);
cfg2.srcdir = [0 0 1 nan]; % isotropic emission
cfg2.e0 = 1; % skip tsearchn/widefield-source setup, unused by elembary
cfg2.prop = [0 0 1 1                    % 0: background/ambient
             muaBackground+muaIcgEm 0 0 1.33]; % 1: ICG cuvette medium at 820nm (real re-absorption)
cfg2.outputtype = 'flux';
cfg2.basisorder = 1;

fluxem = mmclab(cfg2);
fluxemcw = sum(fluxem.data, 2) * cfg2.tstep;

%% plot the estimated fluorescence emission fluence

figure;
qmeshcut(cfg2.elem(cfg2.elemprop > 0, 1:4), cfg2.node, log10(fluxemcw), 'x=5', 'linestyle', 'none');
view([1 0 0]);
box on;
axis equal;
title('estimated 820nm fluorescence emission fluence (300nM ICG cuvette)');
colorbar;
colormap(jet);
