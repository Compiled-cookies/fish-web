$(window).load(function(){ 
	$('#cont_compressed_blocks').masonry({
		itemSelector: '.block_to_compress',
		singleMode: false,
		isResizable: true,
		isAnimated: true,
		animationOptions: { 
			queue: false, 
			duration: 500 
		}
	});
});
