//for canvas drawing used code from here: https://github.com/zealerww/digits_recognition/blob/master/digits_recognition/static/draw.js
var drawing = false;
var context;
var offset_left = 0;
var offset_top = 0;
var canvas_size = 260;
var eraser_active = false;
var linewidth = 4

function start_canvas () {
    var canvas = document.getElementById ("Canvas");
    context = canvas.getContext ("2d");
    canvas.onmousedown = function (event) {mousedown(event)};
    canvas.onmousemove = function (event) {mousemove(event)};
    canvas.onmouseup   = function (event) {mouseup(event)};
    for (var o = canvas; o ; o = o.offsetParent) {
    offset_left += (o.offsetLeft - o.scrollLeft);
    offset_top  += (o.offsetTop - o.scrollTop);
    }
    draw();
}

function getPosition(evt) {
    evt = (evt) ?  evt : ((event) ? event : null);
    var left = 0;
    var top = 0;
    var canvas = document.getElementById("Canvas");

    if (evt.pageX) {
    left = evt.pageX;
    top  = evt.pageY;
    } else if (document.documentElement.scrollLeft) {
    left = evt.clientX + document.documentElement.scrollLeft;
    top  = evt.clientY + document.documentElement.scrollTop;
    } else  {
    left = evt.clientX + document.body.scrollLeft;
    top  = evt.clientY + document.body.scrollTop;
    }
    left -= offset_left;
    top -= offset_top;

    return {x : left, y : top}; 
}

function
mousedown(event) {
    drawing = true;
    var location = getPosition(event);
    if (eraser_active) {
        context.strokeStyle="#ffffff";
        context.lineWidth = 3 * linewidth;
    }
    else {
        context.strokeStyle="#000000";
        context.lineWidth = linewidth;
    }
    context.beginPath();
    context.moveTo(location.x,location.y);
}

function
mousemove(event) {
    if (!drawing) 
        return;
    var location = getPosition(event);
    context.lineTo(location.x,location.y);
    context.stroke();
}


function
mouseup(event) {
    if (!drawing) 
        return;
    mousemove(event);
	context.closePath();
    drawing = false;
}

function draw() {
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas_size, canvas_size);
}


function clear_canvas() {
    document.getElementById('get_ideas_header').innerHTML = '';
    document.getElementById('get_ideas_result').innerHTML = '';
    document.getElementById("prediction_result").innerHTML = '';
    context.clearRect (0, 0, canvas_size, canvas_size);
    draw();
}

function start_drawing() {
    eraser_active = false;
}

function start_erasing() {
    eraser_active = true;
}

onload = start_canvas;