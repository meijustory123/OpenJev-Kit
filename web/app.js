"use strict";
const $ = (id) => document.getElementById(id);
let sessionToken = "", modelReady = false, busy = false, mode = "form", stopped = false, lastResponse = null;
const modelId = "openjev-qwen3.5-0.8b-v1";
const samples = {
  intent: {model:modelId,state:"客户：昨天我说想退款，现在改主意了。鞋子本身没问题，只是尺码偏小，请换成大一码的。退款就不要办理了。",questions:{"当前诉求":{type:"choice",instructions:"识别客户本人最后明确提出且未撤回的处理意图。",criteria:{"换货":"明确要求更换商品或规格","退款":"明确要求退回货款","咨询":"仅询问条件，未提出办理请求"}}}},
  emotion: {model:modelId,state:"小周评价这次住宿：房间安静，床也舒服，这两点我很喜欢；但早餐总是补得太慢，让我有些失望。同伴说以后不来了，不过那不是我的决定。",questions:{"住宿感受":{type:"choice",instructions:"只判断小周本人对本次整体住宿体验的情感，区分他人态度。",criteria:{"正面":"只有正面评价","负面":"只有负面评价","混合":"同时明确表达正面与负面评价","中性":"仅作事实陈述，未表达褒贬","信息不足":"缺少本人针对目标的可判断表达"}}}},
  mixed: {model:modelId,state:{"反馈":"收到的包颜色发错了，请换成下单时选的棕色。我不要求退款。","订单":"已签收；未提供是否拆除吊牌的信息。"},questions:{"处理方式":{type:"choice",instructions:"客户当前明确要求哪种处理？",criteria:{"换货":"更换商品或规格","退款":"退回款项","维修":"修复商品"}},"诉求完整度":{type:"score",instructions:"只评价是否说明问题及期望处理方式。",criteria:["未说明具体问题","说明问题但未明确处理方式","问题和处理方式均明确"]},"要求退款":{type:"noul",instructions:"客户当前是否明确要求退款？"}}}
};
function error(message="") { $("input-error").textContent=message; $("input-error").hidden=!message; }
function updateSubmit() { $("submit").disabled=!modelReady||busy||stopped; $("submit-text").textContent=busy?"正在判断…":"开始判断"; }
function updateCriteria() {
  const type=$("type").value; $("criteria-field").hidden=type==="noul";
  $("criteria-label").textContent=type==="score"?"有序等级（从低到高）":"候选选项";
  $("criteria-help").textContent=type==="score"?"每行一个等级，按从低到高排列；需要 2–10 个等级。":"每行一项，格式：名称 | 描述。名称不能重复。";
}
function formRequest() {
  const state=$("state").value.trim(), name=$("question-id").value.trim(), instructions=$("instructions").value.trim(), type=$("type").value;
  if(!state||!name||!instructions) throw Error("请填写当前状态、问题名称和判断说明。");
  const question={type,instructions};
  if(type!=="noul") {
    const lines=$("criteria").value.split("\n").map(x=>x.trim()).filter(Boolean);
    if(type==="score") { if(lines.length<2||lines.length>10) throw Error("评分需要 2–10 个有序等级。"); question.criteria=lines; }
    else { if(!lines.length||lines.length>255) throw Error("请填写 1–255 个候选选项。");
      question.criteria=Object.create(null);
      for(const line of lines) { const parts=line.split(/[|｜]/), key=parts.shift().trim();
        if(!key) throw Error("每个选项都需要名称。");
        if(Object.hasOwn(question.criteria,key)) throw Error(`选项名称重复：${key}`);
        question.criteria[key]=parts.join(" | ").trim()||null;
      }
    }
  }
  return {model:modelId,state,questions:{[name]:question}};
}
function setMode(next) {
  if(next===mode)return;
  if(next==="json") { try { $("request-json").value=JSON.stringify(formRequest(),null,2); } catch(e) {error(e.message);return;} }
  if(next==="form") { try {
    const value=JSON.parse($("request-json").value), entries=Object.entries(value.questions||{});
    if(typeof value.state!=="string"||entries.length!==1) throw Error("结构化状态或多个问题请继续使用 JSON 模式，避免丢失内容。");
    fillForm(value);
  } catch(e) {error(e.message);return;} }
  applyMode(next);error();
}
function applyMode(next) {mode=next;$("form-panel").hidden=next!=="form";$("json-panel").hidden=next!=="json";$("form-tab").setAttribute("aria-selected",String(next==="form"));$("json-tab").setAttribute("aria-selected",String(next==="json"));}
function fillForm(value) {
  const [name,q]=Object.entries(value.questions)[0];
  if(typeof q.instructions!=="string")throw Error("结构化说明请使用 JSON 模式。");
  if(q.type==="choice"&&Object.values(q.criteria).some(x=>typeof x!=="string"&&x!==null))throw Error("结构化选项请使用 JSON 模式。");
  if(q.type==="score"&&q.criteria.some(x=>typeof x!=="string"))throw Error("结构化等级请使用 JSON 模式。");
  if(q.type==="noul"&&q.criteria)throw Error("含自定义是非定义，请使用 JSON 模式。");
  $("state").value=value.state;$("question-id").value=name;$("type").value=q.type;$("instructions").value=q.instructions;
  $("criteria").value=q.type==="choice"?Object.entries(q.criteria).map(([k,v])=>`${k} | ${v||""}`).join("\n"):(q.criteria||[]).join("\n");updateCriteria();
}
async function api(path,body,raw=false) {
  const response=await fetch(path,{method:body===undefined?"GET":"POST",headers:{"Content-Type":"application/json","X-OpenJev-Token":sessionToken},body:body===undefined?undefined:(raw?body:JSON.stringify(body))});
  const value=await response.json();
  if(!response.ok)throw Error(typeof value.detail==="string"?value.detail:JSON.stringify(value.detail||value));
  return value;
}
async function refreshStatus() {
  if(stopped)return;
  try {
    const s=await api("/api/status");modelReady=s.status==="ready";
    $("status-title").textContent={waiting:"等待微调模型",loading:"正在加载模型",ready:"模型已就绪",error:"模型需要处理"}[s.status]||s.status;
    $("status-dot").className="status-dot "+(modelReady?"ready":s.status==="error"?"error":"");
    $("status-message").textContent=s.message;
    $("model-detail").textContent=s.checkpoint?`${s.checkpoint} · ${s.device==="cuda"?"显卡运行":"CPU运行"}`:"只加载已完整保存的微调模型";
    const t=s.training;
    $("training-progress").textContent=t.active?`训练进行中 · 第 ${t.step.toLocaleString()} / ${(t.planned_steps||"—").toLocaleString()} 步`:(t.step?`最近训练进度 · 第 ${t.step.toLocaleString()} 步`:"本机服务已启动");
    if(t.active&&s.device==="cpu")$("training-progress").textContent+=" · 自动避开训练使用的显卡";
    $("reload").disabled=!s.latest_checkpoint||s.status==="loading"||busy;
    $("reload").textContent=s.checkpoint&&s.latest_checkpoint!==s.checkpoint?"有新断点 · 加载":"加载最新模型";
    if(s.max_length)$("input-hint").textContent=`每个候选输入最多 ${s.max_length.toLocaleString()} token`;
  } catch(e) {modelReady=false;$("status-title").textContent="服务连接中断";$("status-message").textContent="请重新双击启动工具，或等待本机服务恢复。";$("status-dot").className="status-dot error";$("reload").disabled=true;}
  updateSubmit();
}
function element(tag,cls,text) {const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;}
function probabilityRow(label,value,best) {
  const row=element("div","probability"),head=element("div","probability-label"+(best?" best":"")),track=element("progress","probability-track"+(best?" best":""));
  head.append(element("span","",label),element("span","",`${(value*100).toFixed(1)}%`));
  track.max=1;track.value=Math.max(0,Math.min(1,value));track.setAttribute("aria-label",`${label}：${(value*100).toFixed(1)}%`);row.append(head,track);return row;
}
function render(value,elapsed) {
  $("results").replaceChildren();$("result-empty").hidden=true;
  for(const [name,answer] of Object.entries(value.answers)) {
    const card=element("article","answer"),heading=element("div","answer-heading");heading.append(element("h3","",name),element("span","type-tag",{choice:"分类",score:"评分",noul:"是非"}[answer.type]));card.append(heading);
    if(answer.type==="choice") {card.append(element("div","verdict",answer.choice));for(const [key,p] of Object.entries(answer.probabilities))card.append(probabilityRow(key,p,key===answer.choice));}
    else if(answer.type==="score") {card.append(element("div","verdict",`${answer.score.toFixed(2)} / ${Object.keys(answer.legend).length-1}`));for(const [key,p] of Object.entries(answer.probabilities))card.append(probabilityRow(`${key} · ${answer.legend[key]}`,p,p===Math.max(...Object.values(answer.probabilities))));}
    else {card.append(element("div","verdict",`命题成立 ${(answer.noul*100).toFixed(1)}%`));card.append(probabilityRow("是",answer.noul,answer.noul>=.5),probabilityRow("否",1-answer.noul,answer.noul<.5));}
    $("results").append(card);
  }
  lastResponse=value;$("response-json").textContent=JSON.stringify(value,null,2);$("raw-details").hidden=false;$("result-meta").hidden=false;$("token-count").textContent=`累计处理 ${value.usage.input_tokens.toLocaleString()} token`;
  $("timing").textContent=`用时 ${elapsed.toFixed(1)} 秒`;
}
$("form-tab").onclick=()=>setMode("form");$("json-tab").onclick=()=>setMode("json");
$("type").onchange=()=>{if($("type").value==="score")$("criteria").value="未说明具体问题\n说明问题但未明确处理方式\n问题和处理方式均明确";else if($("type").value==="choice")$("criteria").value="选项一 | 此选项的适用范围\n选项二 | 此选项的适用范围";updateCriteria();};
document.querySelectorAll("[data-example]").forEach(button=>button.onclick=()=>{if(busy)return;const value=samples[button.dataset.example];if(button.dataset.example==="mixed"){applyMode("json");$("request-json").value=JSON.stringify(value,null,2);}else{fillForm(value);applyMode("form");}error();});
$("submit").onclick=async()=>{
  error();let value;try{value=mode==="form"?formRequest():JSON.parse($("request-json").value);}catch(e){error(e.message);return;}
  busy=true;updateSubmit();$("reload").disabled=true;$("results").replaceChildren();$("raw-details").hidden=true;$("result-meta").hidden=true;$("result-empty").hidden=true;$("result-loading").hidden=false;$("timing").textContent="";
  const jsonMode=mode==="json", payload=jsonMode?$("request-json").value:value;
  const started=performance.now();try{render(await api("/v1/systemone",payload,jsonMode),(performance.now()-started)/1000);}catch(e){error(e.message);$("result-empty").hidden=false;}finally{busy=false;$("result-loading").hidden=true;updateSubmit();refreshStatus();}
};
$("reload").onclick=async()=>{error();$("reload").disabled=true;modelReady=false;updateSubmit();try{await api("/api/reload",{device:$("device").value});}catch(e){error(e.message);}refreshStatus();};
$("copy").onclick=async()=>{try{await navigator.clipboard.writeText($("response-json").textContent);$("copy").textContent="已复制";setTimeout(()=>$("copy").textContent="复制 JSON",1800);}catch{error("复制失败，可以直接选中完整响应文本复制。");}};
$("download").onclick=()=>{if(!lastResponse)return;const url=URL.createObjectURL(new Blob([JSON.stringify(lastResponse,null,2)],{type:"application/json"})),link=document.createElement("a");link.href=url;link.download="openjev-result.json";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$("shutdown").onclick=async()=>{if(!confirm("关闭网页模型服务并释放它占用的内存？训练任务会继续运行。"))return;try{await api("/api/shutdown",{});stopped=true;modelReady=false;updateSubmit();$("reload").disabled=true;$("status-title").textContent="工具已退出";$("status-message").textContent="可以关闭此页面，下次双击启动工具即可再次使用。";}catch(e){error(e.message);}};
fillForm(samples.intent);
(async()=>{try{sessionToken=(await api("/api/session")).token;await refreshStatus();}catch(e){error("无法连接本机服务，请重新打开工具。");}setInterval(refreshStatus,3000);})();
