$ ->
  $(".input-group").keypress (k) ->
    if k.which == 13 # enter key pressed
      $("#summarize_button").click()
      return false;
  $("#summarize_button").click -> do_summary()


$('#isFull').click -> $(this).toggleClass 'checked'


do_summary = () ->
  query = $("#query_text").val()
  start = $("#start").val()
  isFull = false


  if $('#isFull').hasClass 'checked'
    isFull = true

  if query.length != 0
    $("#search_results_list").empty()
    $.ajax "get-pdf",
      type: "POST"
      contentType: "application/json; charset=utf-8"
      dataType: "json"
      data: JSON.stringify
        query: query,
        start: start,
        isFull: isFull
      success: (data, stat, xhr) -> print_results data
      failure: (axhr, stat, err) ->
        $("#search_results_list").append("<li>Something bad happened!</li>")



print_results = (result) ->
  console.log(result)
  html = "<p>#{result["introduction"]}"
  $("#search_results_list").append(html)
