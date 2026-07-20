"""Shared machinery for the interactive overhead-track charts on the dashboard pages
(the vio.html 3-lap Win chart and the nav.html Stage-5b showdown).

One self-contained canvas renderer (TRACKCHART_JS, no external libraries) consumed by both
page generators:

    scripts/embed_vio_win.py    splices data + chart script into the hand-authored vio.html
    scripts/build_dashboard.py  renders the nav.html Stage-5b chart at build time

The engine draws equal-aspect XY tracks on a 1 m grid with numbered waypoint squares, a start
marker, final-position X marks, nearest-point hover markers, and legend buttons that toggle
series *groups* (a series is visible iff every group it belongs to is on — that is how the
Stage-5b "believed paths" master toggle composes with the per-run toggles). Everything
page-specific — the series, groups, and readout formatters — stays in the page generator and
is passed in via the cfg object.
"""
import base64
import io
import re

from PIL import Image


def png_b64(path):
    """Optimize + flatten alpha (same treatment as the pages' other embedded PNGs)."""
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def swap_img(html, img_id, path):
    pat = re.compile(r'(<img id="' + img_id + r'"[^>]*?src="data:image/png;base64,)[^"]*(")')
    n = len(pat.findall(html))
    if n != 1:
        raise SystemExit(f"expected exactly one img id={img_id!r} with a base64 src, found {n}")
    return pat.sub(lambda m: m.group(1) + png_b64(path) + m.group(2), html)


def splice(html, marker, block):
    """Replace the single <!-- {marker}_START --> ... <!-- {marker}_END --> region with block."""
    pat = re.compile("<!-- " + marker + "_START.*?" + marker + "_END -->", re.DOTALL)
    n = len(pat.findall(html))
    if n != 1:
        raise SystemExit(f"expected exactly one {marker} block, found {n}")
    return pat.sub(lambda m: block, html)


def chart_css(ns):
    """The chart's CSS for one id namespace (#{ns}wrap/chart/legend/read/fallback) — mirrors the
    hand-authored #win* rules in vio.html so both charts read as the same instrument."""
    return f"""
/* interactive track chart — canvas shown by JS; #{ns}fallback serves no-JS AND print */
#{ns}wrap{{display:none;background:var(--inset);border:1px solid var(--border);border-radius:4px;padding:10px 12px}}
#{ns}chart{{display:block;width:100%;height:560px;cursor:crosshair}}
@media (max-width:900px){{#{ns}chart{{height:420px}}}}
#{ns}legend{{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:9px}}
#{ns}legend button{{
  display:inline-flex;align-items:center;gap:6px;font-size:9.5px;letter-spacing:.1em;
  text-transform:uppercase;padding:3px 9px;border:1px solid var(--border2);border-radius:3px;
  background:rgba(255,255,255,.015);font-family:var(--mono);cursor:pointer;
}}
#{ns}legend button .sw{{width:14px;height:3px;border-radius:2px;display:inline-block}}
#{ns}legend button.off{{opacity:.35}}
#{ns}legend button.off span:last-child{{text-decoration:line-through}}
#{ns}read{{margin-top:9px;font-size:10.5px;color:var(--mut);letter-spacing:.03em;min-height:17px;font-variant-numeric:tabular-nums}}
@media print{{#{ns}wrap{{display:none !important}}#{ns}fallback{{display:block !important}}}}
"""


# The shared renderer. cfg:
#   wrap/canvas/legend/readout/fallback  element ids (fallback hidden once the canvas is live)
#   series     [{pts:[[x,y]..], color, dash?, lw?, alpha?, groups:[key..], scan?, mark?}]
#              drawn in order; scan → participates in hover; mark → final-position X
#   groups     [{key, label, color, on?}] → legend buttons; series visible iff all its groups on
#   waypoints  [[x,y]..] numbered grey squares (optional)
#   start      {p:[x,y], label} green start dot (optional)
#   idle       string (or function(on)) shown when not hovering
#   fmt        function(seriesIdx, ptIdx, on) → readout HTML while hovering; the hovered series
#              gets a ring, every other visible scan series a dot at the same index (clamped to
#              its own length, so runs of different durations stay comparable)
TRACKCHART_JS = """
function TrackChart(cfg){
  var cv=document.getElementById(cfg.canvas),wrap=document.getElementById(cfg.wrap),
      fb=document.getElementById(cfg.fallback),rd=document.getElementById(cfg.readout),
      lg=document.getElementById(cfg.legend);
  if(!cv||!cv.getContext)return;                 // no canvas: the PNG fallback stands
  wrap.style.display="block";if(fb)fb.style.display="none";
  var ctx=cv.getContext("2d"),SER=cfg.series,WP=cfg.waypoints||[],hov=null,on={};
  cfg.groups.forEach(function(g){on[g.key]=g.on!==false;});
  function vis(s){return s.groups.every(function(k){return on[k];});}
  var xs=[],ys=[];
  SER.forEach(function(s){for(var i=0;i<s.pts.length;i++){xs.push(s.pts[i][0]);ys.push(s.pts[i][1]);}});
  WP.forEach(function(w){xs.push(w[0]);ys.push(w[1]);});
  var x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs),
      y0=Math.min.apply(null,ys),y1=Math.max.apply(null,ys),W,H,S,ox,oy;
  function layout(){
    var w=cv.clientWidth,h=cv.clientHeight,dpr=window.devicePixelRatio||1;
    cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);W=w;H=h;
    S=Math.min(w/(x1-x0),h/(y1-y0))*0.93;
    ox=(w-S*(x1-x0))/2;oy=(h-S*(y1-y0))/2;
  }
  function px(p){return[ox+S*(p[0]-x0),H-(oy+S*(p[1]-y0))];}
  function line(pts,color,dash,lw,alpha){
    ctx.save();ctx.strokeStyle=color;ctx.lineWidth=lw||1.4;
    if(dash)ctx.setLineDash(dash);
    if(alpha!==undefined)ctx.globalAlpha=alpha;
    ctx.beginPath();
    for(var i=0;i<pts.length;i++){var q=px(pts[i]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);}
    ctx.stroke();ctx.restore();
  }
  function draw(){
    ctx.clearRect(0,0,W,H);
    ctx.save();ctx.strokeStyle="rgba(63,208,201,.07)";ctx.lineWidth=1;    // 1 m grid
    for(var gx=Math.ceil(x0);gx<=x1;gx++){var a=px([gx,y0]),b=px([gx,y1]);
      ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
    for(var gy=Math.ceil(y0);gy<=y1;gy++){var c=px([x0,gy]),d=px([x1,gy]);
      ctx.beginPath();ctx.moveTo(c[0],c[1]);ctx.lineTo(d[0],d[1]);ctx.stroke();}
    ctx.restore();
    SER.forEach(function(s){if(vis(s))line(s.pts,s.color,s.dash,s.lw,s.alpha);});
    WP.forEach(function(w,k){                                             // numbered waypoint squares
      var q=px(w);ctx.fillStyle="#6b7686";ctx.fillRect(q[0]-7.5,q[1]-7.5,15,15);
      ctx.fillStyle="#f2f5f9";ctx.font="9px ui-monospace,monospace";ctx.textAlign="center";
      ctx.fillText(String(k+1),q[0],q[1]+3);ctx.textAlign="start";
    });
    if(cfg.start){
      var s0=px(cfg.start.p);
      ctx.fillStyle="#4ade80";ctx.beginPath();ctx.arc(s0[0],s0[1],4.5,0,6.2832);ctx.fill();
      ctx.fillStyle="#8a95a5";ctx.font="10px ui-monospace,monospace";
      ctx.fillText(cfg.start.label,s0[0]+9,s0[1]+4);
    }
    SER.forEach(function(s){                                              // final positions: x marks
      if(!s.mark||!vis(s))return;var q=px(s.pts[s.pts.length-1]);
      ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();
      ctx.moveTo(q[0]-5,q[1]-5);ctx.lineTo(q[0]+5,q[1]+5);
      ctx.moveTo(q[0]+5,q[1]-5);ctx.lineTo(q[0]-5,q[1]+5);ctx.stroke();
    });
    if(hov){
      SER.forEach(function(s,si){
        if(!s.scan||!vis(s))return;
        var i=Math.min(hov.i,s.pts.length-1),q=px(s.pts[i]);
        if(si===hov.s){ctx.strokeStyle=s.color;ctx.lineWidth=1.2;
          ctx.beginPath();ctx.arc(q[0],q[1],5,0,6.2832);ctx.stroke();}
        else{ctx.fillStyle=s.color;ctx.beginPath();ctx.arc(q[0],q[1],3.5,0,6.2832);ctx.fill();}
      });
    }
  }
  function readout(){
    rd.innerHTML=hov?cfg.fmt(hov.s,hov.i,on):(typeof cfg.idle==="function"?cfg.idle(on):cfg.idle);
  }
  cv.addEventListener("mousemove",function(e){
    var r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,best=1600,bs=-1,bi=-1;
    SER.forEach(function(s,si){
      if(!s.scan||!vis(s))return;
      for(var i=0;i<s.pts.length;i++){var q=px(s.pts[i]),
        dd=(q[0]-mx)*(q[0]-mx)+(q[1]-my)*(q[1]-my);if(dd<best){best=dd;bs=si;bi=i;}}
    });
    hov=bi<0?null:{s:bs,i:bi};draw();readout();
  });
  cv.addEventListener("mouseleave",function(){hov=null;draw();readout();});
  cfg.groups.forEach(function(g){
    var b=document.createElement("button");
    b.innerHTML="<span class=\\"sw\\" style=\\"background:"+g.color+"\\"></span><span style=\\"color:"+
      g.color+"\\">"+g.label+"</span>";
    if(g.on===false)b.classList.add("off");
    b.addEventListener("click",function(){on[g.key]=!on[g.key];
      b.classList.toggle("off",!on[g.key]);hov=null;draw();readout();});
    lg.appendChild(b);
  });
  layout();draw();readout();
  window.addEventListener("resize",function(){layout();draw();});
}
"""
