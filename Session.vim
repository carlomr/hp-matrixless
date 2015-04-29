let SessionLoad = 1
if &cp | set nocp | endif
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Documents/LJLL/deal.II-dev/examples/hp-matrixless
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +2 operators.hpp
badd +1 operators.cpp
badd +194 ~/Documents/Polimi/PACS/TSPEED/lib/include/QuadratureRule.hpp
badd +23 ~/Documents/Polimi/PACS/TSPEED/lib/include/QuadratureRule_imp.hpp
badd +1 boundary_tentativo
badd +1 ~/Documents/LJLL/deal.II-dev/examples/assembled_hp-modified/CMakeLists.txt
badd +69 ~/Documents/LJLL/codice_mio/lib/include/Mesh.hpp
badd +201 ~/Documents/LJLL/codice_mio/lib/include/Geometry.hpp
badd +24 ../../../../Polimi/PACS/TSPEED/lib/src/ShapeFunctions.cpp
badd +1 test_one_dim_mat.cpp
badd +6 Bases.hpp
badd +241 ~/Documents/LJLL/deal.II-dev/examples/hp-matrixless/bases.hpp
badd +17 test_simple_laplacian.cpp
args operators.hpp
edit ~/Documents/LJLL/deal.II-dev/examples/hp-matrixless/bases.hpp
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winheight=1 winwidth=1
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
29,37fold
16,39fold
58,62fold
54,66fold
52,75fold
42,78fold
101,114fold
81,116fold
10,118fold
132,152fold
174,175fold
169,176fold
190,199fold
187,200fold
156,203fold
156,203fold
213,217fold
219,224fold
231,233fold
228,234fold
226,235fold
207,237fold
241,252fold
10
normal! zo
16
normal! zo
16
normal! zc
42
normal! zo
52
normal! zo
54
normal! zo
54
normal! zc
52
normal! zc
42
normal! zc
81
normal! zo
81
normal! zc
10
normal! zc
132
normal! zo
156
normal! zo
156
normal! zo
169
normal! zo
174
normal! zo
187
normal! zo
190
normal! zo
156
normal! zc
207
normal! zo
226
normal! zo
228
normal! zo
228
normal! zc
226
normal! zc
241
normal! zo
let s:l = 236 - ((87 * winheight(0) + 24) / 48)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
236
normal! 015|
tabnext 1
if exists('s:wipebuf')
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToO
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
let g:this_obsession = v:this_session
unlet SessionLoad
" vim: set ft=vim :
