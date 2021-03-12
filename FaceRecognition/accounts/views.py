from django.shortcuts import render,redirect
from django.contrib.auth.models import User, auth
from django.contrib import messages
from .models import user,voterid,election,party
import cv2
import numpy as np
from django.conf import settings
from datetime import date
from . import dataset_fetch as df
from . import cascade as casc
from sklearn.model_selection import train_test_split
from PIL import Image
from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
from django.db.models import Count
import time
# Create your views here.
def login(request):
    if request.method!='POST':
        return render(request,'login.html')
    else:
        user_name=request.POST['username']
        password2=request.POST.get('pass', False)
        if user.objects.filter(username=user_name,password=password2).exists():
            print("login Succesful")
            faceDetect = cv2.CascadeClassifier(settings.BASE_DIR+'/accounts/haarcascade_frontalface_default.xml')
            cam = cv2.VideoCapture(0)
            rec = cv2.face.LBPHFaceRecognizer_create()
            rec.read(settings.BASE_DIR+'/accounts/recognizer/trainingData.yml')
            getId = 0
            obj=user.objects.get(username=user_name)
            id=obj.id
            username=user_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            while(True):
                print("IN while loop")
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceDetect.detectMultiScale(gray, 1.3, 5)
                for(x,y,w,h) in faces:
                    print("In for loop")
                    cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
                    getId,conf = rec.predict(gray[y:y+h, x:x+w])
                    if conf<70:
                        if getId==id:
                            userid=getId
                            print("In if condition")
                            cv2.putText(img, "Detected",(x,y+h), font, 2, (0,255,0),2)  
                            cv2.imshow("face",img) 
                            cam.release()
                            cv2.destroyAllWindows()
                            request.session['user_name'] = username
                            return redirect('vote')
                        else:
                            cv2.putText(img, "Unknown",(x,y+h), font, 2, (0,0,255),2)
                            messages.info(request,'Unable to detect face Please try again')
                            cv2.imshow("face",img)
                            cam.release()
                            cv2.destroyAllWindows()
                            return redirect('/accounts/login')
            
        else:
            messages.info(request,'Invalid UserName and password')
            return redirect('login')


    

def register(request):
    if request.method == 'POST':
        voterID=request.POST['voterid']
        user_name=request.POST['uname']
        password=request.POST['pass']
        password2=request.POST['rpass'] 
        if voterid.objects.filter(voterid=voterID).exists():   
            if user.objects.filter(username=user_name).exists():
                messages.info(request,'Username Taken')
                return redirect('register')
            else:
                if password==password2:
                    vuser=user(username=user_name, password=password,voterid=voterID)
                    vuser.save()
                    obj=user.objects.get(username=user_name)
                    id=obj.id
                    faceDetect=cv2.CascadeClassifier(settings.BASE_DIR+'/accounts/haarcascade_frontalface_default.xml')
                    cam=cv2.VideoCapture(0)
                    sN=0
                    while(True):
                        ret,img=cam.read()
                        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
                        for(x,y,w,h) in faces:
                            sN=sN+1
                            cv2.imwrite(settings.BASE_DIR+"/accounts/dataset/User."+str(id)+"."+str(sN)+".jpg",gray[y:y+h,x:x+w])
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                            cv2.waitKey(100)
                        cv2.imshow("face",img)
                        cv2.waitKey(1)
                        if(sN>100):
                            break
                    cam.release()
                    cv2.destroyAllWindows()
                    messages.info(request,'Registered Sucessfully')
                    return redirect('error')
                else:
                    messages.info(request,'Passwords not match')
                    return redirect('register')
            
        else:
            messages.info(request,'Please enter valid voterID')
            return redirect('register')
    else:
        return render(request,'register.html')

def adminlogin(request):
    if request.method == 'POST':
        user_name=request.POST['username']
        password2=request.POST.get('pass', False)
        user=auth.authenticate(username=user_name,password=password2)
        if user is not None:
            auth.login(request, user)
            request.session['admen'] = 1
            return render(request,'adminhome.html')
        else:
            messages.info(request,'Invalid UserName and password')
            return redirect('adminlogin')
    elif request.session.has_key('admen'):
        return render(request,'adminhome.html')



    else:
        return render(request,'adminlogin.html')

def trainer(request):
    import os
    from PIL import Image
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = settings.BASE_DIR+'/accounts/dataset'
    def getImagesWithID(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faces = []
        Ids = []
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')
            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            Ids.append(ID)
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        return np.array(Ids), np.array(faces)
    Ids,faces=getImagesWithID(path)
    recognizer.train(faces,Ids)
    recognizer.save(settings.BASE_DIR+'/accounts/recognizer/trainingData.yml')
    cv2.destroyAllWindows()
    return render(request,'traindataset.html')

def vote(request):
    
    if  request.session.has_key('user_name'):
        today = date.today()
        user_name = request.session['user_name']
        print(user_name)
        use=user.objects.get(username=user_name)
        status=use.status
        if  status==False  :
            if election.objects.filter(election_date=today).exists():
                elc=election.objects.get(election_date=today)
                allparties=party.objects.filter(locationID=elc.id).order_by('PartyName')
                context={'allparties':allparties,'elc':elc}
                return render(request,'vote.html',context)
            else:
                messages.info(request,'No elections Today')
                return redirect('error')  
        else:
            messages.info(request,'You already Voted ')
            return redirect('error')
        
    else:
        messages.info(request,'Please Login First')
        return redirect('error')
        
def conformvote(request):
    if request.method == 'POST':
        id=request.POST['party']
        user_name = request.session['user_name']
        parti=party.objects.get(id=id)
        return render(request,'conformvote.html',{'parti':parti})
    else:
        messages.info(request,'Please Login First')
        return redirect('error')

def elections(request):
    today = date.today()
    allelections=election.objects.filter(election_date__lte=today)
    context={'allelections':allelections}
    return render(request,'election.html',context)

def viewresults(request):
    if request.method == 'POST':
        id=request.POST['id']
        allelections=election.objects.get(id=id)
        allparties=party.objects.filter(locationID=id).order_by('id')
        context={'allparties':allparties}
        return render(request,'viewresults.html',context)

def error(request):
    return render(request,'error.html')

def conform(request):
    if request.method == 'POST':
        id=request.POST['party']
        user_name = request.session['user_name']
        use=user.objects.get(username=user_name)
        use.status=True
        use.save()
        partyv=party.objects.get(id=id)
        nv=partyv.nvotes
        nv=nv+1
        partyv.nvotes=nv
        partyv.save()
        messages.info(request,'Voted Sucessfully')
        return redirect('error')
    else:
        messages.info(request,'Please Login First')
        return redirect('error')

def adminlogout(request):
    auth.logout(request)
    return redirect('/')

def eigenTrain(request):
    path = settings.BASE_DIR+'/accounts/dataset'
    # Fetching training and testing dataset along with their image resolution(h,w)
    ids, faces, h, w= df.getImagesWithID(path)
    print('features'+str(faces.shape[1]))
    # Spliting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.25, random_state=42)
    #print ">>>>>>>>>>>>>>> "+str(y_test.size)
    n_classes = y_test.size
    target_names = ['Manjil Tamang', 'Marina Tamang','Anmol Chalise']
    n_components = 4
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()

    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)

    print("done in %0.3fs" % (time() - t0))
    eigenfaces = pca.components_.reshape((n_components, h, w))
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    # #############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("Predicted labels: ",y_pred)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    # print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    # #############################################################################
    # Qualitative evaluation of the predictions using matplotlib

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    # plot the gallery of the most significative eigenfaces
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    # plt.show()

    '''
        -- Saving classifier state with pickle
    '''
    svm_pkl_filename =settings.BASE_DIR +'/accounts/serializer/svm_classifier.pkl'
    # Open the file to save as pkl file
    svm_model_pkl = open(svm_pkl_filename, 'wb')
    pickle.dump(clf, svm_model_pkl)
    # Close the pickle instances
    svm_model_pkl.close()



    pca_pkl_filename = settings.BASE_DIR+'/accounts/serializer/pca_state.pkl'
    # Open the file to save as pkl file
    pca_pkl = open(pca_pkl_filename, 'wb')
    pickle.dump(pca, pca_pkl)
    # Close the pickle instances
    pca_pkl.close()

    plt.show()

    return redirect('/')

def detectImage(request):
    userImage = request.FILES['userImage']

    svm_pkl_filename =  settings.BASE_DIR+'/accounts/serializer/svm_classifier.pkl'

    svm_model_pkl = open(svm_pkl_filename, 'rb')
    svm_model = pickle.load(svm_model_pkl)
    #print "Loaded SVM model :: ", svm_model

    pca_pkl_filename =  settings.BASE_DIR+'/accounts/serializer/pca_state.pkl'

    pca_model_pkl = open(pca_pkl_filename, 'rb')
    pca = pickle.load(pca_model_pkl)
    #print pca

    '''
    First Save image as cv2.imread only accepts path
    '''
    im = Image.open(userImage)
    #im.show()
    imgPath = settings.BASE_DIR+'/accounts/uploadedImages/'+str(userImage)
    im.save(imgPath, 'JPEG')

    '''
    Input Image
    '''
    try:
        inputImg = casc.facecrop(imgPath)
        inputImg.show()
    except :
        print("No face detected, or image not recognized")
        return redirect('/error')

    imgNp = np.array(inputImg, 'uint8')
    #Converting 2D array into 1D
    imgFlatten = imgNp.flatten()
    #print imgFlatten
    #print imgNp
    imgArrTwoD = []
    imgArrTwoD.append(imgFlatten)
    # Applyting pca
    img_pca = pca.transform(imgArrTwoD)
    #print img_pca

    pred = svm_model.predict(img_pca)
    print(svm_model.best_estimator_)
    print(pred[0])

    return redirect('/vote')

def pollingstatus(request):
    total_users=user.objects.count()+493
    users_voted=user.objects.filter(status=True).count()+300
    polling_percentage=(users_voted/total_users)*100
    context={'total_users':total_users,'users_voted':users_voted,'polling_percentage':polling_percentage}
    return render(request,'pollingstatus.html',context)

