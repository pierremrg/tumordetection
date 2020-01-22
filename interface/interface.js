$(function(){

	var classifiers = ['knn', 'gbc', 'rfc', 'svm', 'mlp', 'resnet', 'alexnet', 'vgg', 'cnnmedium'];
	var classifiers_names = ['k-NN', 'GBC', 'RFC', 'SVM', 'MLP', 'ResNet', 'AlexNet', 'VGG', 'CNN (Medium)'];

	function getClassifierName(classifier){
		return classifiers_names[classifiers.indexOf(classifier)];
	}

	// Let's train
	$('#form-train').on('submit', function(){
		var checked_classifiers = [];

		for(var i=0; i<classifiers.length; i++){
			if($('#' + classifiers[i]).is(":checked"))
				checked_classifiers.push(classifiers[i]);
		}

		var url_db = $('#db_url').val();

		var data = {
			url_db: url_db,
			classifiers: checked_classifiers
		};

		$.ajax({
			url: 'simu_orchestrateur.php',
			type: 'POST',
			data: data,
			dataType: 'json',
			success: function(data){
				$('#train-section .results').empty();

				$.each(data.returns_trains, function(classifier, values){

					$('#train-section .results').append(
						'<div class="result-classifier">'+
						'<p class="classifier">'+getClassifierName(classifier)+'</p>' +
						'<p class="small">Train accuracy: '+parseInt(values.train_acc*100)+'%'+
						'<div class="progress">' +
						'<div class="complete" ' +
						'style="width:'+parseInt(values.train_acc*100)+'%' +
						';background:#00A4DF"></div>' +
						'</div>'+
						'<p class="small">Validation accuracy: '+parseInt(values.val_acc*100)+'%'+
						'<div class="progress">' +
						'<div class="complete" ' +
						'style="width:'+parseInt(values.val_acc*100)+'%' +
						';background:#00A4DF"></div>' +
						'</div>'
					);

				});

				$('#train-section .results').fadeIn();

			},
			error: function(response, statut, error){
				alert('Error: ' + error)
			}
		});

		return false;
	});


	// File input
	$('#loadpicture').click(function(){
		$('#picture').click();
		return false;
	});

	var ajaxPicture;
	$('#picture').change(function(){
		ajaxPicture = new FormData();

		ajaxPicture.append('picture', $(this)[0].files[0]);

		$('#lets-classify').attr('disabled', false);
		
		return false;

	});

	$('#lets-classify').click(function(){

		var checked_classifiers = [];

		for(var i=0; i<classifiers.length; i++){
			if($('#test-' + classifiers[i]).is(":checked"))
				checked_classifiers.push(classifiers[i]);
		}

		ajaxPicture.append('classifiers', checked_classifiers);

		$.ajax({
			url: 'http://localhost:5003/api/v1/simu_orchestrateur',
			type: 'POST',
			data: ajaxPicture,
			dataType: 'json',
			cache: false,
			contentType: false,
			processData: false,
			success: function(data){

				if(data === null){
					alert('Error during analysis.');
					return false;
				}

				$('#test-section .results').empty();

				$.each(data.returns_predictions, function(classifier, values){

					var result;
					if(values.label)
						result = 'yes';
					else
						result = 'no';

					$('#test-section .results').append(
						'<div class="result-classifier">'+
						'<p class="classifier">'+getClassifierName(classifier)+
						'<span class="icon '+result+'">'+result+'</span>'+
						'</p><p class="small">Probability: '+parseInt(values.proba*100)+'%'+
						'<div class="progress">' +
						'<div class="complete" ' +
						'style="width:'+parseInt(values.proba*100)+'%' +
						';background:#D62246"></div>' +
						'</div>'
					);

					$('#test-section .results').fadeIn();

				});

			},
			error: function(response, statut, error){
				alert('Error: ' + error);
			}
		});

		return false;

	});


});