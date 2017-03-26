// alert("Hello, world!")

// Takes the first hyperreference on a given page and
// shows it on the console

// var firstHref = $("a[href^='http']").eq(0).attr("href");
// console.log(firstHref);

chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse){
        if(request.message == "clicked_browser_action"){
            var firstHref = $("a[href^='http']").eq(0).attr("href");

            console.log(firstHref);
        }
    }
);