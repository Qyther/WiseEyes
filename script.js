const networkSize=[224,224];
var inputs=[];
var outputs=[];
var images=[];
const classes=[];
var imageNetwork;
var collecting;
var running=false;
var predicting=false;
var firstPressed=0;




addEventListener("load",()=>{
    const status=document.getElementsByClassName("scriptStatus")[0];
    const camera=document.getElementsByClassName("camera")[0];

    function dataGather() {
        collecting=!collecting?this.getAttribute("data-index"):false;
        if(!firstPressed++) status.innerText="Parsing camera...";
        setTimeout(dataGatherLoop);
    }
    const accessCam=()=>{
        if(running) return;
        if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia) return status.innerText="Your device is not supported.";
        navigator.mediaDevices.getUserMedia({video:true,width:640,height:480}).then(stream=>{
            camera.srcObject=stream;
            camera.addEventListener("loadeddata",()=>{
                running=true;
                document.getElementsByClassName("entryScreen")[0].style.display="none";
            });
        });
    };
    const dataGatherLoop=()=>{
        if(!running||!collecting) return;
        let cleanImage=tf.tidy(()=>imageNetwork.predict(tf.image.resizeBilinear(tf.browser.fromPixels(camera),networkSize.reverse(),true).div(255).expandDims()).squeeze());
        inputs.push(cleanImage);
        outputs.push(collecting);
        if(!images[collecting]) images[collecting]=0;
        images[collecting]++;
        status.innerText="";
        for(let i=0;i<classes.length;i++) status.innerText+=(i%2===0||i===0?"":", ")+classes[i]+" - "+(images[i]?images[i]:0)+" images"+(i%2===1?"\n":"");
        requestAnimationFrame(dataGatherLoop);
    };
    var reset=false;
    var lastTime=[];
    var predictionTreshold=2500;
    const predictLoop=()=>{
        if(!predicting) return;
        tf.tidy(()=>{
            let prediction=model.predict(imageNetwork.predict(tf.image.resizeBilinear(tf.browser.fromPixels(camera).div(255),networkSize.reverse(),true).expandDims())).squeeze();
            let predictionArray=prediction.arraySync();
            let predictionIndex=prediction.argMax().arraySync();
            if(predictionIndex===4) {
                reset=true;
                lastTime=[];
            } else if(reset&&lastTime[0]!==predictionIndex) lastTime=[predictionIndex,Date.now()];
            else if(reset&&Date.now()-lastTime[1]>predictionTreshold) {
                updateView(predictionIndex);
                updateMenu();
                reset=false;
                lastTime=[];
            }
            if(lastTime.length===0) status.innerText=classes[predictionIndex]+", confidence "+(predictionArray[predictionIndex]*100).toFixed(1)+"%";
            else status.innerText=classes[predictionIndex]+", confidence "+(predictionArray[predictionIndex]*100).toFixed(1)+"%, "+((Date.now()-lastTime[1])/1000).toFixed(1)+"s/"+predictionTreshold/1000+"s";
        });
        requestAnimationFrame(predictLoop);
    };
    const readyAll=async()=>{
        status.innerText="Preparing to learn...";
        predict=false;
        tf.util.shuffleCombo(inputs,outputs);
        let output=tf.tensor1d(outputs,"int32");
        let bitOutputs=tf.oneHot(output,classes.length);
        let input=tf.stack(inputs);
        const epochCount=10;
        setTimeout(async()=>{
            await model.fit(input,bitOutputs,{shuffle:true,batchSize:8,epochs:epochCount,callbacks:{onEpochEnd:epoch=>status.innerText="Training: "+(epoch/epochCount*100).toFixed(1)+"%"}});
            output.dispose();
            bitOutputs.dispose();
            input.dispose();
            predicting=true;
            document.getElementsByClassName("setupButtons")[0].style.display="none";
            document.getElementsByClassName("selectionMenu")[0].style.display="block";
            predictLoop();
        });
    };

    var currentView=options;
    const updateMenu=()=>{
        let keys=Object.keys(currentView);
        for(let i=0;i<4;i++) {
            document.getElementsByClassName("navigationText")[i].innerText=keys[i];
        }
    };
    updateMenu();
    const updateView=a=>{
        let newView=currentView[Object.keys(currentView)[a]];
        if(newView!==0&&newView!==-1) return currentView=newView;
        if(newView===0) {
            let TTS=new SpeechSynthesisUtterance();
            TTS.text=Object.keys(currentView)[a].toLowerCase();
            TTS.lang="en-US";
            speechSynthesis.speak(TTS);
        }
        currentView=options;
        updateMenu();
    };


    document.documentElement.addEventListener("mousedown",accessCam);
    document.getElementsByClassName("train")[0].addEventListener("click",readyAll);
    document.getElementsByClassName("reset")[0].addEventListener("click",()=>{
        predicting=false;
        inputs.map(x=>x.dispose());
        inputs.length=outputs.length=images.length=0;
        status.innerText="No data.";
    });
    document.getElementsByClassName("back")[0].addEventListener("click",()=>{
        document.getElementsByClassName("setupButtons")[0].style.display="block";
        document.getElementsByClassName("selectionMenu")[0].style.display="none";
    });

    let dataButtons=document.getElementsByClassName("dataButton");
    for(let i=0;i<dataButtons.length;i++) {
        dataButtons[i].addEventListener("ontouchstart" in document?"touchstart":"mousedown",dataGather);
        dataButtons[i].addEventListener("ontouchend" in document?"touchend":"mouseup",dataGather);
        classes.push(dataButtons[i].getAttribute("class-name"));
    }

    (async()=>{
        try {
            const URL="https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";
            imageNetwork=await tf.loadGraphModel(URL,{fromTFHub:true});
            status.innerText="Resources loaded.";
        } catch(e) {
            status.innerText=e;
        }
    })();

    let model=tf.sequential();
    model.add(tf.layers.dense({inputShape:[1024],units:128,activation:"relu"}));
    model.add(tf.layers.dense({units:classes.length,activation:"softmax"}));
    model.summary();
    model.compile({optimizer:"adam",loss:"categoricalCrossentropy",metrics:["accuracy"]});
});


const options={
    "Animals/Things":{
        Return:-1,
        Animals:{
            Return:-1,
            Farm:{
                Return:-1,
                Cow:0,
                Pig:0,
                "None of the above":{
                    Return:-1,
                    Chicken:0,
                    Sheep:0,
                    "None of the above":{
                        Return:-1,
                        Bee:0,
                        Donkey:0,
                        Horse:0
                    }
                }
            },
            Pets:{
                Return:-1,
                Cat:0,
                Dog:0,
                "None of the above":{
                    Return:-1,
                    Bird:0,
                    Hamster:0,
                    Rabbit:0
                }
            },
            Other:{
                Return:-1,
                Insect:{
                    Return:-1,
                    Ant:0,
                    Bug:0,
                    Spider:0
                },
                Mammal:{
                    Return:-1,
                    Carnivore:{
                        Return:-1,
                        Tiger:0,
                        Lion:0,
                        Wolf:0
                    },
                    Herbivore:{
                        Return:-1,
                        Deer:0,
                        Elephant:0,
                        Giraffe:0
                    },
                    Omnivore:{
                        Return:-1,
                        Bear:0,
                        Hippo:0,
                        "None of the above":{
                            Return:-1,
                            Fox:0,
                            Monkey:0,
                            Rat:0
                        }
                    }
                },
                Reptile:{
                    Return:-1,
                    Tortoise:0,
                    Turtle:0,
                    "None of the above":{
                        Return:-1,
                        Crocodile:0,
                        Lizard:0,
                        Snake:0
                    }
                }
            }
        },
        Food:{
            Return:-1,
            Solids:{
                Ingredients:{
                    Return:-1,
                    Carbs:{
                        Return:-1,
                        Potatoes:0,
                        Sugar:0,
                        "None of the above":{
                            Return:-1,
                            Beans:0,
                            Bread:0,
                            Rice:0
                        }
                    },
                    Protein:{
                        Return:-1,
                        Beef:0,
                        Dairy:{
                            Return:-1,
                            Cheese:0,
                            Milk:0,
                            Yogurt:0
                        },
                        Pork:0
                    },
                    "Vegetables/Nuts/Fruit":{
                        Return:-1,
                        Fruit:{
                            Return:-1,
                            Apple:0,
                            Banana:0,
                            "None of the above":{
                                Return:-1,
                                Raspberry:0,
                                Strawberry:0,
                                "None of the above":{
                                    Return:-1,
                                    Grape:0,
                                    Melon:0,
                                    "None of the above":{
                                        Return:-1,
                                        Orange:0,
                                        Pineapple:0,
                                        "None of the above":{
                                            Return:-1,
                                            Berry:0,
                                            Kiwi:0,
                                            Lemon:0
                                        }
                                    }
                                }
                            }
                        },
                        Nuts:{
                            Return:-1,
                            Almond:0,
                            Peanut:0,
                            "None of the above":{
                                Return:-1,
                                Walnut:0,
                                Cashew:0,
                                Pistachio:0
                            }
                        },
                        Vegetables:{
                            Return:-1,
                            Carrot:0,
                            Eggplant:0,
                            "None of the above":{
                                Return:-1,
                                Lettuce:0,
                                Tomato:0,
                                "None of the above":{
                                    Return:-1,
                                    Cauliflower:0,
                                    Corn:0,
                                    Spinach:0
                                }
                            }
                        }
                    }
                },
                Meal:{
                    Return:-1,
                    Breakfast:{
                        Return:-1,
                        Cornflakes:0,
                        Muesli:0,
                        "None of the above":{
                            Return:-1,
                            Bread:0,
                            Waffle:0,
                            "None of the above":{
                                Return:-1,
                                Cracker:0,
                                Pancake:0,
                                "None of the above":{
                                    Return:-1,
                                    Baguette:0,
                                    Croissant:0,
                                    Yogurt:0
                                }
                            }
                        }
                    },
                    Lunch:{
                        Return:-1,
                        Bagel:0,
                        Salad:0,
                        Sandwich:0
                    },
                    Dinner:{
                        Return:-1,
                        Pasta:0,
                        Pizza:0,
                        "None of the above":{
                            Return:-1,
                            Lasagna:0,
                            Soup:0,
                            "None of the above":{
                                Return:-1,
                                Steak:0,
                                Stew:0,
                                Quiche:0
                            }
                        }
                    },
                },
                Snacks:{
                    Return:-1,
                    Cake:0,
                    Chips:0,
                    Cookie:0
                }
            },
            Drinks:{
                Return:-1,
                Alcohol:{
                    Return:-1,
                    Beer:0,
                    Vodka:0,
                    Wine:0
                },
                Regular:{
                    Return:-1,
                    Coffee:0,
                    Tea:0,
                    "None of the above":{
                        Return:-1,
                        "Chocolate milk":0,
                        Juice:0,
                        Water:0,
                    }
                },
                Soft:{
                    Return:-1,
                    Coke:0,
                    Fanta:0,
                    Sprite:0
                }
            },
            Sauce:{
                Return:-1,
                "BBQ sauce":0,
                Ketchup:0,
                Mayonnaise:0,
            }
        },
        Objects:{
            Return:-1,
            Clothing:{
                Return:-1,
                Pants:0,
                Shirt:0,
                "None of the above":{
                    Return:-1,
                    Shoes:0,
                    Sweater:0,
                    "None of the above":{
                        Return:-1,
                        Blouse:0,
                        Socks:0,
                        Underwear:0
                    }
                }
            },
            Furniture:{
                Return:-1,
                Chair:0,
                Sofa:0,
                "None of the above":{
                    Return:-1,
                    Bed:0,
                    Desk:0,
                    "None of the above":{
                        Return:-1,
                        Armchair:0,
                        Bookshelf:0,
                        Table:0
                    }
                }
            },
            "None of the above":{
                Return:-1,
                Instruments:{
                    Return:-1,
                    Guitar:0,
                    Piano:0,
                    "None of the above":{
                        Return:-1,
                        Cello:0,
                        Violin:0,
                        "None of the above":{
                            Return:-1,
                            Saxophone:0,
                            Trumpet:0,
                            "None of the above":{
                                Return:-1,
                                Bass:0,
                                Drums:0,
                                Trombone:0
                            }
                        }
                    }
                },
                Useful:{
                    Return:-1,
                    Backpack:0,
                    Pencil:0,
                    "None of the above":{
                        Return:-1,
                        Book:0,
                        "Pencil Case":0,
                        Notebook:0
                    }
                },
                "None of the above":{
                    Return:-1,
                    "Tools/Weapons":{
                        Return:-1,
                        Drill:0,
                        Hammer:0,
                        "None of the above":{
                            Return:-1,
                            Axe:0,
                            Sword:0,
                            "None of the above":{
                                Return:-1,
                                Gun:0,
                                Saw:0,
                                Screwdriver:0
                            }
                        }
                    },
                    Vehicles:{
                        Return:-1,
                        Car:0,
                        Truck:0,
                        "None of the above":{
                            Return:-1,
                            Boat:0,
                            Plane:0,
                            Train:0
                        }
                    },
                    "None of the above":{
                        Return:-1,
                        Electronics:{
                            Return:-1,
                            Computer:0,
                            Smartphone:0,
                            "None of the above":{
                                Return:-1,
                                Headphones:0,
                                Keyboard:0,
                                "None of the above":{
                                    Return:-1,
                                    Microphone:0,
                                    Mouse:0,
                                    Tablet:0
                                }
                            }
                        },
                        Materials:{
                            Return:-1,
                            Metal:0,
                            Plastic:0,
                            Wood:0
                        },
                        Plants:{
                            Return:-1,
                            Cactus:0,
                            Bush:0,
                            Tree:0
                        }
                    }
                }
            }
        }
    },
    General:{
        Return:-1,
        "Adjectives/Adverbs":{
            Return:-1,
            Adjectives:{
                Return:-1,
                Positive:{
                    Return:-1,
                    Charming:0,
                    Fantastic:0,
                    "None of the above":{
                        Return:-1,
                        Gentle:0,
                        Perfect:0,
                        "None of the above":{
                            Return:-1,
                            Good:0,
                            Tasty:0,
                            Eager:0
                        }
                    }
                },
                Negative:{
                    Return:-1,
                    Bad:0,
                    Rough:0,
                    Sharp:0
                },
                Neutral:{
                    Return:-1,
                    Big:0,
                    Small:0,
                    "None of the above":{
                        Return:-1,
                        Round:0,
                        Square:0,
                        Sharp:0
                    }
                }
            },
            Adverbs:{
                Return:-1,
                Not:0,
                Also:0,
                "None of the above":{
                    Return:-1,
                    Too:0,
                    Very:0,
                    "None of the above":{
                        Return:-1,
                        Never:0,
                        Sometimes:0,
                        "None of the above":{
                            Return:-1,
                            Again:0,
                            Alone:0,
                            Together:0,
                        }
                    }
                }
            },
            The:0
        },
        Conjunctions:{
            Return:-1,
            And:0,
            Because:0,
            "None of the above":{
                Return:-1,
                But:0,
                Unlike:0,
                "None of the above":{
                    Return:-1,
                    Since:0,
                    Unless:0,
                    "None of the above":{
                        Return:-1,
                        If:0,
                        So:0,
                        "None of the above":{
                            Return:-1,
                            After:0,
                            Before:0,
                            Yet:0
                        }
                    }
                }
            }
        },
        Prepositions:{
            Return:-1,
            But:0,
            Like:0,
            "None of the above":{
                Return:-1,
                Because:0,
                Before:0,
                "None of the above":{
                    Return:-1,
                    Above:0,
                    In:0,
                    Under:0
                }
            }
        }
    },
    Person:{
        Return:-1,
        Body:{
            Return:-1,
            Upper:{
                Return:-1,
                Head:{
                    Return:-1,
                    Ears:0,
                    Eyes:0,
                    "None of the above":{
                        Return:-1,
                        Hair:0,
                        Mouth:0,
                        Nose:0
                    }
                },
                Neck:{
                    Return:-1,
                    "Adam's apple":0,
                    Chin:0,
                    Throat:0
                },
                Organs:{
                    Return:-1,
                    Brain:0,
                    Skin:0,
                    Tongue:0
                }
            },
            Middle:{
                Return:-1,
                Arms:{
                    Return:-1,
                    Elbow:0,
                    Finger:0,
                    Hand:0
                },
                Back:0,
                Chest:{
                    Return:-1,
                    Bosom:0,
                    Stomach:0,
                    Organs:{
                        Return:-1,
                        Heart:0,
                        Stomach:0,
                        "None of the above":{
                            Return:-1,
                            Bowels:0,
                            Kidney:0,
                            "None of the above":{
                                Return:-1,
                                Bladder:0,
                                Liver:0,
                                Lungs:0
                            }
                        }
                    }
                }
            },
            Lower:{
                Return:-1,
                Hips:0,
                Feet:{
                    Return:-1,
                    Ankle:0,
                    Heel:0,
                    Toes:0,
                },
                Legs:{
                    Return:-1,
                    Calf:0,
                    Knee:0,
                    Thigh:0
                }
            }
        },
        Emotions:{
            Return:-1,
            Positive:{
                Return:-1,
                Content:0,
                Happy:0,
                Pleased:0
            },
            Negative:{
                Return:-1,
                Sad:0,
                Angry:0,
                Disgusted:0
            },
            Neutral:{
                Return:-1,
                Hungry:0,
                Shy:0,
                Surprised:0
            }
        },
        Pronouns:{
            Return:-1,
            Me:{
                Return:-1,
                I:0,
                Me:0,
                "None of the above":{
                    Return:-1,
                    My:0,
                    Mine:0,
                    "None of the above":{
                        Return:-1,
                        We:0,
                        Us:0,
                        Our:0,
                    }
                }
            },
            You:{
                Return:-1,
                You:0,
                Your:0,
                Yours:0
            },
            They:{
                Return:-1,
                He:0,
                Him:0,
                "None of the above":{
                    Return:-1,
                    His:0,
                    She:0,
                    "None of the above":{
                        Return:-1,
                        Her:0,
                        Hers:0,
                        "None of the above":{
                            Return:-1,
                            They:0,
                            Them:0,
                            Their:0
                        }
                    }
                }
            }
        }
    },
    Verb:{
        Return:-1,
        Action:{
            Return:-1,
            Mental:{
                Return:-1,
                "A to I":{
                    Return:-1,
                    Concern:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Concerned:0,
                            "Was concerned":0,
                            "Had concerned":0
                        },
                        Present:{
                            Return:-1,
                            Concern:0,
                            Concerns:0,
                            Concerning:0
                        },
                        Future:{
                            Return:-1,
                            "Will concern":0,
                            "Will be concerning":0,
                            "Will have concerned":0
                        }
                    },
                    Decide:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Decided:0,
                            "Was deciding":0,
                            "Had decided":0
                        },
                        Present:{
                            Return:-1,
                            Decide:0,
                            Decides:0,
                            Deciding:0
                        },
                        Future:{
                            Return:-1,
                            "Will decide":0,
                            "Will be deciding":0,
                            "Will have decided":0
                        }
                    },
                    "None of the above":{
                        Return:-1,
                        Dislike:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Disliked:0,
                                "Was disliking":0,
                                "Had disliked":0
                            },
                            Present:{
                                Return:-1,
                                Dislike:0,
                                Dislikes:0,
                                Disliking:0
                            },
                            Future:{
                                Return:-1,
                                "Will dislike":0,
                                "Will be disliking":0,
                                "Will have disliked":0
                            }
                        },
                        Doubt:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Doubted:0,
                                "Was doubting":0,
                                "Had doubted":0
                            },
                            Present:{
                                Return:-1,
                                Doubt:0,
                                Doubts:0,
                                Doubting:0
                            },
                            Future:{
                                Return:-1,
                                "Will doubt":0,
                                "Will be doubting":0,
                                "Will have doubted":0
                            }
                        },
                        "None of the above":{
                            Return:-1,
                            Hear:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Heard:0,
                                    "Was hearing":0,
                                    "Had heard":0
                                },
                                Present:{
                                    Return:-1,
                                    Hear:0,
                                    Hears:0,
                                    Hearing:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will hear":0,
                                    "Will be hearing":0,
                                    "Will have heard":0
                                }
                            },
                            Hope:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Hoped:0,
                                    "Was hoping":0,
                                    "Had hoped":0
                                },
                                Present:{
                                    Return:-1,
                                    Hope:0,
                                    Hopes:0,
                                    Hoping:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will hope":0,
                                    "Will be hoping":0,
                                    "Will have hoped":0
                                }
                            },
                            "None of the above":{
                                Return:-1,
                                Impress:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Impressed:0,
                                        "Was impressing":0,
                                        "Had impressed":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Impress:0,
                                        Impresses:0,
                                        Impressing:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will impress":0,
                                        "Will be impressing":0,
                                        "Will have impressed":0
                                    }
                                },
                                Ignore:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Ignored:0,
                                        "Was ignoring":0,
                                        "Had ignored":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Ignore:0,
                                        Ignores:0,
                                        Ignoring:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will ignore":0,
                                        "Will be ignoring":0,
                                        "Will have ignored":0
                                    }
                                },
                                Imagine:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Imagined:0,
                                        "Was imagining":0,
                                        "Had imagined":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Imagine:0,
                                        Imagines:0,
                                        Imagining:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will imagine":0,
                                        "Will be imagining":0,
                                        "Will have imagined":0
                                    }
                                }
                            }
                        }
                    }
                },
                "J to Q":{
                    Return:-1,
                    Know:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Knew:0,
                            "Was knowing":0,
                            "Had known":0
                        },
                        Present:{
                            Return:-1,
                            Know:0,
                            Knows:0,
                            Knowing:0
                        },
                        Future:{
                            Return:-1,
                            "Will know":0,
                            "Will be knowing":0,
                            "Will have known":0
                        }
                    },
                    Learn:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Learned:0,
                            "Was learning":0,
                            "Had learned":0
                        },
                        Present:{
                            Return:-1,
                            Learn:0,
                            Learns:0,
                            Learning:0
                        },
                        Future:{
                            Return:-1,
                            "Will learn":0,
                            "Will be learning":0,
                            "Will have learned":0
                        }
                    },
                    "None of the above":{
                        Return:-1,
                        Like:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Liked:0,
                                "Was liking":0,
                                "Had liked":0
                            },
                            Present:{
                                Return:-1,
                                Like:0,
                                Likes:0,
                                Liking:0
                            },
                            Future:{
                                Return:-1,
                                "Will like":0,
                                "Will be liking":0,
                                "Will have liked":0
                            }
                        },
                        Look:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Looked:0,
                                "Was looking":0,
                                "Had looked":0
                            },
                            Present:{
                                Return:-1,
                                Look:0,
                                Looks:0,
                                Looking:0
                            },
                            Future:{
                                Return:-1,
                                "Will look":0,
                                "Will be looking":0,
                                "Will have looked":0
                            }
                        },
                        "None of the above":{
                            Return:-1,
                            Love:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Loved:0,
                                    "Was loving":0,
                                    "Had loved":0
                                },
                                Present:{
                                    Return:-1,
                                    Love:0,
                                    Loves:0,
                                    Loving:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will love":0,
                                    "Will be loving":0,
                                    "Will have loved":0
                                }
                            },
                            Mind:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Minded:0,
                                    "Was minding":0,
                                    "Had minded":0
                                },
                                Present:{
                                    Return:-1,
                                    Mind:0,
                                    Minds:0,
                                    Minding:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will mind":0,
                                    "Will be minding":0,
                                    "Will have minded":0
                                }
                            },
                            "None of the above":{
                                Return:-1,
                                Notice:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Noticed:0,
                                        "Was noticing":0,
                                        "Had noticed":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Notice:0,
                                        Notices:0,
                                        Noticing:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will notice":0,
                                        "Will be noticing":0,
                                        "Will have noticed":0
                                    }
                                },
                                Own:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Owned:0,
                                        "Was owning":0,
                                        "Had owned":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Own:0,
                                        Owns:0,
                                        Owning:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will own":0,
                                        "Will be owning":0,
                                        "Will have owned":0
                                    }
                                },
                                "None of the above":{
                                    Return:-1,
                                    Perceive:{
                                        Return:-1,
                                        Past:{
                                            Return:-1,
                                            Perceived:0,
                                            "Was perceiving":0,
                                            "Had perceived":0
                                        },
                                        Present:{
                                            Return:-1,
                                            Perceive:0,
                                            Perceives:0,
                                            Perceiving:0
                                        },
                                        Future:{
                                            Return:-1,
                                            "Will perceive":0,
                                            "Will be perceiving":0,
                                            "Will have perceived":0
                                        }
                                    },
                                    Prefer:{
                                        Return:-1,
                                        Past:{
                                            Return:-1,
                                            Preferred:0,
                                            "Was preferring":0,
                                            "Had preferred":0
                                        },
                                        Present:{
                                            Return:-1,
                                            Prefer:0,
                                            Prefers:0,
                                            Preferring:0
                                        },
                                        Future:{
                                            Return:-1,
                                            "Will prefer":0,
                                            "Will be preferring":0,
                                            "Will have preferred":0
                                        }
                                    },
                                    Pretend:{
                                        Return:-1,
                                        Past:{
                                            Return:-1,
                                            Pretended:0,
                                            "Was pretending":0,
                                            "Had pretended":0
                                        },
                                        Present:{
                                            Return:-1,
                                            Pretend:0,
                                            Pretends:0,
                                            Pretending:0
                                        },
                                        Future:{
                                            Return:-1,
                                            "Will pretend":0,
                                            "Will be pretending":0,
                                            "Will have pretended":0
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "R to Z":{
                    Return:-1,
                    Realize:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Realized:0,
                            "Was realizing":0,
                            "Had realized":0
                        },
                        Present:{
                            Return:-1,
                            Realize:0,
                            Realizes:0,
                            Realizing:0
                        },
                        Future:{
                            Return:-1,
                            "Will realize":0,
                            "Will be realizing":0,
                            "Will have realized":0
                        }
                    },
                    Recognize:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Recognized:0,
                            "Was recognizing":0,
                            "Had recognized":0
                        },
                        Present:{
                            Return:-1,
                            Recognize:0,
                            Recognizes:0,
                            Recognizing:0
                        },
                        Future:{
                            Return:-1,
                            "Will recognize":0,
                            "Will be recognizing":0,
                            "Will have recognized":0
                        }
                    },
                    "None of the above":{
                        Return:-1,
                        "Remember":{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Remembered:0,
                                "Was remembering":0,
                                "Had remembered":0
                            },
                            Present:{
                                Return:-1,
                                Remember:0,
                                Remembers:0,
                                Remembering:0
                            },
                            Future:{
                                Return:-1,
                                "Will remember":0,
                                "Will be remembering":0,
                                "Will have remembered":0
                            }
                        },
                        "See":{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Saw:0,
                                "Was seeing":0,
                                "Had seen":0
                            },
                            Present:{
                                Return:-1,
                                See:0,
                                Sees:0,
                                Seeing:0
                            },
                            Future:{
                                Return:-1,
                                "Will see":0,
                                "Will be seeing":0,
                                "Will have seen":0
                            }
                        },
                        "None of the above":{
                            Return:-1,
                            Smell:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Smelled:0,
                                    "Was smelling":0,
                                    "Had smelled":0
                                },
                                Present:{
                                    Return:-1,
                                    Smell:0,
                                    Smells:0,
                                    Smelling:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will smell":0,
                                    "Will be smelling":0,
                                    "Will have smelled":0
                                }
                            },
                            Surprise:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Surprised:0,
                                    "Was surprising":0,
                                    "Had surprised":0
                                },
                                Present:{
                                    Return:-1,
                                    Surprise:0,
                                    Surprises:0,
                                    Surprising:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will surprise":0,
                                    "Will be surprising":0,
                                    "Will have surprised":0
                                }
                            },
                            "None of the above":{
                                Return:-1,
                                Think:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Thought:0,
                                        "Was thinking":0,
                                        "Had thought":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Think:0,
                                        Thinks:0,
                                        Thinking:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will think":0,
                                        "Will be thinking":0,
                                        "Will have thought":0
                                    }
                                },
                                Translate:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Translated:0,
                                        "Was translating":0,
                                        "Had translated":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Translate:0,
                                        Translates:0,
                                        Translating:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will translate":0,
                                        "Will be translating":0,
                                        "Will have translated":0
                                    }
                                },
                                Understand:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Understood:0,
                                        "Was understanding":0,
                                        "Had understood":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Understand:0,
                                        Understands:0,
                                        Understanding:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will understand":0,
                                        "Will be understanding":0,
                                        "Will have understood":0
                                    }
                                }
                            }
                        }
                    }
                }
            },
            Physical:{
                Return:-1,
                "A to D":{
                    Return:-1,
                    Act:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Acted:0,
                            "Was acting":0,
                            "Had acted":0
                        },
                        Present:{
                            Return:-1,
                            Act:0,
                            Acts:0,
                            Acting:0
                        },
                        Future:{
                            Return:-1,
                            "Will act":0,
                            "Will be acting":0,
                            "Will have acted":0
                        }
                    },
                    Answer:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Answered:0,
                            "Was answering":0,
                            "Had answered":0
                        },
                        Present:{
                            Return:-1,
                            Answer:0,
                            Answers:0,
                            Answering:0
                        },
                        Future:{
                            Return:-1,
                            "Will answer":0,
                            "Will be answering":0,
                            "Will have answered":0
                        }
                    },
                    "None of the above":{
                        Return:-1,
                        Approve:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Approved:0,
                                "Was approving":0,
                                "Had approved":0
                            },
                            Present:{
                                Return:-1,
                                Approve:0,
                                Approves:0,
                                Approving:0
                            },
                            Future:{
                                Return:-1,
                                "Will approve":0,
                                "Will be approving":0,
                                "Will have approved":0
                            }
                        },
                        Break:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Broke:0,
                                "Was breaking":0,
                                "Had broken":0
                            },
                            Present:{
                                Return:-1,
                                Break:0,
                                Breaks:0,
                                Breaking:0
                            },
                            Future:{
                                Return:-1,
                                "Will break":0,
                                "Will be breaking":0,
                                "Will have broken":0
                            }
                        },
                        "None of the above":{
                            Return:-1,
                            Build:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Built:0,
                                    "Was building":0,
                                    "Had built":0
                                },
                                Present:{
                                    Return:-1,
                                    Build:0,
                                    Builds:0,
                                    Building:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will build":0,
                                    "Will be building":0,
                                    "Will have built":0
                                }
                            },
                            Buy:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Bought:0,
                                    "Was buying":0,
                                    "Had bought":0
                                },
                                Present:{
                                    Return:-1,
                                    Buy:0,
                                    Buys:0,
                                    Buying:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will buy":0,
                                    "Will be buying":0,
                                    "Will have bought":0
                                }
                            },
                            "None of the above":{
                                Return:-1,
                                Cough:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Coughed:0,
                                        "Was coughing":0,
                                        "Had coughed":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Cough:0,
                                        Coughs:0,
                                        Coughing:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will cough":0,
                                        "Will be coughing":0,
                                        "Will have coughed":0
                                    }
                                },
                                Create:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Created:0,
                                        "Was creating":0,
                                        "Had created":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Create:0,
                                        Creates:0,
                                        Creating:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will create":0,
                                        "Will be creating":0,
                                        "Will have created":0
                                    }
                                },
                                "None of the above":{
                                    Return:-1,
                                    Cry:{
                                        Return:-1,
                                        Past:{
                                            Return:-1,
                                            Cried:0,
                                            "Was crying":0,
                                            "Had cried":0
                                        },
                                        Present:{
                                            Return:-1,
                                            Cry:0,
                                            Cries:0,
                                            Crying:0
                                        },
                                        Future:{
                                            Return:-1,
                                            "Will cry":0,
                                            "Will be crying":0,
                                            "Will have cried":0
                                        }
                                    },
                                    Describe:{
                                        Return:-1,
                                        Past:{
                                            Return:-1,
                                            Described:0,
                                            "Was describing":0,
                                            "Had described":0
                                        },
                                        Present:{
                                            Return:-1,
                                            Describe:0,
                                            Describes:0,
                                            Describing:0
                                        },
                                        Future:{
                                            Return:-1,
                                            "Will describe":0,
                                            "Will be describing":0,
                                            "Will have described":0
                                        }
                                    },
                                    "None of the above":{
                                        Return:-1,
                                        Draw:{
                                            Return:-1,
                                            Past:{
                                                Return:-1,
                                                Drew:0,
                                                "Was drawing":0,
                                                "Had drawn":0
                                            },
                                            Present:{
                                                Return:-1,
                                                Draw:0,
                                                Draws:0,
                                                Drawing:0
                                            },
                                            Future:{
                                                Return:-1,
                                                "Will draw":0,
                                                "Will be drawing":0,
                                                "Will have drawn":0
                                            }
                                        },
                                        Drink:{
                                            Return:-1,
                                            Past:{
                                                Return:-1,
                                                Drank:0,
                                                "Was drinking":0,
                                                "Had drank":0
                                            },
                                            Present:{
                                                Return:-1,
                                                Drink:0,
                                                Drinks:0,
                                                Drinking:0
                                            },
                                            Future:{
                                                Return:-1,
                                                "Will drink":0,
                                                "Will be drinking":0,
                                                "Will have drank":0
                                            }
                                        },
                                        Drop:{
                                            Return:-1,
                                            Past:{
                                                Return:-1,
                                                Dropped:0,
                                                "Was dropping":0,
                                                "Had dropped":0
                                            },
                                            Present:{
                                                Return:-1,
                                                Drop:0,
                                                Drops:0,
                                                Dropping:0
                                            },
                                            Future:{
                                                Return:-1,
                                                "Will drop":0,
                                                "Will be dropping":0,
                                                "Will have dropped":0
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "E to L":{
                    Return:-1,
                    Eat:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Ate:0,
                            "Was eating":0,
                            "Had eaten":0
                        },
                        Present:{
                            Return:-1,
                            Eat:0,
                            Eats:0,
                            Eating:0
                        },
                        Future:{
                            Return:-1,
                            "Will eat":0,
                            "Will be eating":0,
                            "Will have eaten":0
                        }
                    },
                    Edit:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Edited:0,
                            "Was editing":0,
                            "Had edited":0
                        },
                        Present:{
                            Return:-1,
                            Edit:0,
                            Edits:0,
                            Editing:0
                        },
                        Future:{
                            Return:-1,
                            "Will edit":0,
                            "Will be editing":0,
                            "Will have edited":0
                        }
                    },
                    "None of the above":{
                        Return:-1,
                        Enter:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Entered:0,
                                "Was entering":0,
                                "Had entered":0
                            },
                            Present:{
                                Return:-1,
                                Enter:0,
                                Enters:0,
                                Entering:0
                            },
                            Future:{
                                Return:-1,
                                "Will enter":0,
                                "Will be entering":0,
                                "Will have entered":0
                            }
                        },
                        Exit:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Exited:0,
                                "Was exiting":0,
                                "Had exited":0
                            },
                            Present:{
                                Return:-1,
                                Exit:0,
                                Exits:0,
                                Exiting:0
                            },
                            Future:{
                                Return:-1,
                                "Will exit":0,
                                "Will be exiting":0,
                                "Will have exited":0
                            }
                        },
                        "None of the above":{
                            Return:-1,
                            Laugh:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Laughed:0,
                                    "Was laughing":0,
                                    "Had laughed":0
                                },
                                Present:{
                                    Return:-1,
                                    Laugh:0,
                                    Laughs:0,
                                    Laughing:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will laugh":0,
                                    "Will be laughing":0,
                                    "Will have laughed":0
                                }
                            },
                            Lie:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Lied:0,
                                    "Was lying":0,
                                    "Had lied":0
                                },
                                Present:{
                                    Return:-1,
                                    Lie:0,
                                    Lies:0,
                                    Lying:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will lie":0,
                                    "Will be lying":0,
                                    "Will have lied":0
                                }
                            },
                            "None of the above":{
                                Return:-1,
                                Listen:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Listened:0,
                                        "Was listening":0,
                                        "Had listened":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Listen:0,
                                        Listens:0,
                                        Listening:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will listen":0,
                                        "Will be listening":0,
                                        "Will have listened":0
                                    }
                                },
                                Lay:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Laid:0,
                                        "Was laying":0,
                                        "Had laid":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Lay:0,
                                        Lays:0,
                                        Laying:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will lay":0,
                                        "Will be laying":0,
                                        "Will have laid":0
                                    }
                                }
                            }
                        }
                    }
                },
                "P to Z":{
                    Return:-1,
                    "P to R":{
                        Return:-1,
                        Paint:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Painted:0,
                                "Was painting":0,
                                "Had painted":0
                            },
                            Present:{
                                Return:-1,
                                Paint:0,
                                Paints:0,
                                Painting:0
                            },
                            Future:{
                                Return:-1,
                                "Will paint":0,
                                "Will be painting":0,
                                "Will have painted":0
                            }
                        },
                        Plan:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Planned:0,
                                "Was planning":0,
                                "Had planned":0
                            },
                            Present:{
                                Return:-1,
                                Plan:0,
                                Plans:0,
                                Planning:0
                            },
                            Future:{
                                Return:-1,
                                "Will plan":0,
                                "Will be planning":0,
                                "Will have planned":0
                            }
                        },
                        "None of the above":{
                            Return:-1,
                            Play:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Played:0,
                                    "Was playing":0,
                                    "Had played":0
                                },
                                Present:{
                                    Return:-1,
                                    Play:0,
                                    Plays:0,
                                    Playing:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will play":0,
                                    "Will be playing":0,
                                    "Will have played":0
                                }
                            },
                            Read:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Read:0,
                                    "Was reading":0,
                                    "Had read":0
                                },
                                Present:{
                                    Return:-1,
                                    Read:0,
                                    Reads:0,
                                    Reading:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will read":0,
                                    "Will be reading":0,
                                    "Will have read":0
                                }
                            },
                            Run:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Ran:0,
                                    "Was running":0,
                                    "Had ran":0
                                },
                                Present:{
                                    Return:-1,
                                    Run:0,
                                    Runs:0,
                                    Running:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will run":0,
                                    "Will be running":0,
                                    "Will have ran":0
                                }
                            }
                        }
                    },
                    "S to T":{
                        Return:-1,
                        Scream:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Screamed:0,
                                "Was screaming":0,
                                "Had screamed":0
                            },
                            Present:{
                                Return:-1,
                                Scream:0,
                                Screams:0,
                                Screaming:0
                            },
                            Future:{
                                Return:-1,
                                "Will scream":0,
                                "Will be screaming":0,
                                "Will have screamed":0
                            }
                        },
                        See:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Saw:0,
                                "Was seeing":0,
                                "Had seen":0
                            },
                            Present:{
                                Return:-1,
                                See:0,
                                Sees:0,
                                Seeing:0
                            },
                            Future:{
                                Return:-1,
                                "Will see":0,
                                "Will be seeing":0,
                                "Will have seen":0
                            }
                        },
                        "None of the above":{
                            Return:-1,
                            Shout:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Shouted:0,
                                    "Was shouting":0,
                                    "Had shouted":0
                                },
                                Present:{
                                    Return:-1,
                                    Shout:0,
                                    Shouts:0,
                                    Shouting:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will shout":0,
                                    "Will be shouting":0,
                                    "Will have shouted":0
                                }
                            },
                            Sing:{
                                Return:-1,
                                Past:{
                                    Return:-1,
                                    Sang:0,
                                    "Was singing":0,
                                    "Had sung":0
                                },
                                Present:{
                                    Return:-1,
                                    Sing:0,
                                    Sings:0,
                                    Singing:0
                                },
                                Future:{
                                    Return:-1,
                                    "Will sing":0,
                                    "Will be singing":0,
                                    "Will have sung":0
                                }
                            },
                            "None of the above":{
                                Return:-1,
                                Sleep:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Slept:0,
                                        "Was sleeping":0,
                                        "Had slept":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Sleep:0,
                                        Sleeps:0,
                                        Sleeping:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will sleep":0,
                                        "Will be sleeping":0,
                                        "Will have slept":0
                                    }
                                },
                                Sneeze:{
                                    Return:-1,
                                    Past:{
                                        Return:-1,
                                        Sneezed:0,
                                        "Was sneezing":0,
                                        "Had sneezed":0
                                    },
                                    Present:{
                                        Return:-1,
                                        Sneeze:0,
                                        Sneezes:0,
                                        Sneezing:0
                                    },
                                    Future:{
                                        Return:-1,
                                        "Will sneeze":0,
                                        "Will be sneezing":0,
                                        "Will have sneezed":0
                                    }
                                },
                                "None of the above":{
                                    Return:-1,
                                    Study:{
                                        Return:-1,
                                        Past:{
                                            Return:-1,
                                            Studied:0,
                                            "Was studying":0,
                                            "Had studied":0
                                        },
                                        Present:{
                                            Return:-1,
                                            Study:0,
                                            Studies:0,
                                            Studying:0
                                        },
                                        Future:{
                                            Return:-1,
                                            "Will study":0,
                                            "Will be studying":0,
                                            "Will have studied":0
                                        }
                                    },
                                    Teach:{
                                        Return:-1,
                                        Past:{
                                            Return:-1,
                                            Taught:0,
                                            "Was teaching":0,
                                            "Had taught":0
                                        },
                                        Present:{
                                            Return:-1,
                                            Teach:0,
                                            Teaches:0,
                                            Teaching:0
                                        },
                                        Future:{
                                            Return:-1,
                                            "Will teach":0,
                                            "Will be teaching":0,
                                            "Will have taught":0
                                        }
                                    },
                                    Touch:{
                                        Return:-1,
                                        Past:{
                                            Return:-1,
                                            Touched:0,
                                            "Was touching":0,
                                            "Had touched":0
                                        },
                                        Present:{
                                            Return:-1,
                                            Touch:0,
                                            Touches:0,
                                            Touching:0
                                        },
                                        Future:{
                                            Return:-1,
                                            "Will touch":0,
                                            "Will be touching":0,
                                            "Will have touched":0
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "W to Z":{
                        Return:-1,
                        Walk:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Walked:0,
                                "Was walking":0,
                                "Had walked":0
                            },
                            Present:{
                                Return:-1,
                                Walk:0,
                                Walks:0,
                                Walking:0
                            },
                            Future:{
                                Return:-1,
                                "Will walk":0,
                                "Will be walking":0,
                                "Will have walked":0
                            }
                        },
                        Win:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Won:0,
                                "Was winning":0,
                                "Had won":0
                            },
                            Present:{
                                Return:-1,
                                Win:0,
                                Wins:0,
                                Winning:0
                            },
                            Future:{
                                Return:-1,
                                "Will win":0,
                                "Will be winning":0,
                                "Will have won":0
                            }
                        },
                        Write:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Wrote:0,
                                "Was writing":0,
                                "Had written":0
                            },
                            Present:{
                                Return:-1,
                                Write:0,
                                Writes:0,
                                Writing:0
                            },
                            Future:{
                                Return:-1,
                                "Will write":0,
                                "Will be writing":0,
                                "Will have written":0
                            }
                        }
                    }
                }
            },
            Empty:-1
        },
        Helping:{
            Return:-1,
            Auxiliaries:{
                Be:{
                    Return:-1,
                    Past:{
                        Return:-1,
                        Was:0,
                        Were:0,
                        Empty:-1
                    },
                    Present:{
                        Return:-1,
                        Am:0,
                        Are:0,
                        Is:0
                    },
                    Future:{
                        Return:-1,
                        "Will be":0,
                        "Will have been":0,
                        Empty:-1
                    }
                },
                Do:{
                    Return:-1,
                    Past:{
                        Return:-1,
                        Did:0,
                        "Was doing":0,
                        "Had done":0
                    },
                    Present:{
                        Return:-1,
                        Do:0,
                        Does:0,
                        Doing:0
                    },
                    Future:{
                        Return:-1,
                        "Will do":0,
                        "Will be doing":0,
                        "Will have done":0
                    }
                },
                Have:{
                    Return:-1,
                    Past:{
                        Return:-1,
                        Had:0,
                        "Was having":0,
                        "Had had":0
                    },
                    Present:{
                        Return:-1,
                        Have:0,
                        Haves:0,
                        Having:0
                    },
                    Future:{
                        Return:-1,
                        "Will have":0,
                        "Will be having":0,
                        "Will have had":0
                    }
                }
            },
            Modals:{
                Return:-1,
                Can:0,
                Could:0,
                "None of the above":{
                    Return:-1,
                    May:0,
                    Might:0,
                    "None of the above":{
                        Return:-1,
                        Shall:0,
                        Should:0,
                        "None of the above":{
                            Return:-1,
                            Will:0,
                            Would:0,
                            "Ought to":0
                        }
                    }
                }
            },
            Empty:-1
        },
        Linking:{
            Return:-1,
            "A to L":{
                Return:-1,
                Be:{
                    Return:-1,
                    Past:{
                        Return:-1,
                        Was:0,
                        Were:0,
                        Empty:-1
                    },
                    Present:{
                        Return:-1,
                        Am:0,
                        Are:0,
                        Is:0
                    },
                    Future:{
                        Return:-1,
                        "Will be":0,
                        "Will have been":0,
                        Empty:-1
                    }
                },
                Appear:{
                    Return:-1,
                    Past:{
                        Return:-1,
                        Appeared:0,
                        "Was appearing":0,
                        "Had appeared":0
                    },
                    Present:{
                        Return:-1,
                        Appear:0,
                        Appears:0,
                        Appearing:0
                    },
                    Future:{
                        Return:-1,
                        "Will appear":0,
                        "Will be appearing":0,
                        "Will have appeared":0
                    }
                },
                "None of the above":{
                    Return:-1,
                    Become:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Became:0,
                            "Was becoming":0,
                            "Had become":0
                        },
                        Present:{
                            Return:-1,
                            Become:0,
                            Becomes:0,
                            Becoming:0
                        },
                        Future:{
                            Return:-1,
                            "Will become":0,
                            "Will be becoming":0,
                            "Will have become":0
                        }
                    },
                    Been:0,
                    "None of the above":{
                        Return:-1,
                        Feel:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Felt:0,
                                "Was feeling":0,
                                "Had felt":0
                            },
                            Present:{
                                Return:-1,
                                Feel:0,
                                Feels:0,
                                Feeling:0
                            },
                            Future:{
                                Return:-1,
                                "Will feel":0,
                                "Will be feeling":0,
                                "Will have felt":0
                            }
                        },
                        Grow:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Grew:0,
                                "Was growing":0,
                                "Had grown":0
                            },
                            Present:{
                                Return:-1,
                                Grow:0,
                                Grows:0,
                                Growing:0
                            },
                            Future:{
                                Return:-1,
                                "Will grow":0,
                                "Will be growing":0,
                                "Will have grown":0
                            }
                        },
                        Look:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Looked:0,
                                "Was looking":0,
                                "Had looked":0
                            },
                            Present:{
                                Return:-1,
                                Look:0,
                                Looks:0,
                                Looking:0
                            },
                            Future:{
                                Return:-1,
                                "Will look":0,
                                "Will be looking":0,
                                "Will have looked":0
                            }
                        }
                    }
                }
            },
            "R to W":{
                Return:-1,
                Remain:{
                    Return:-1,
                    Past:{
                        Return:-1,
                        Remained:0,
                        "Was remaining":0,
                        "Had remained":0
                    },
                    Present:{
                        Return:-1,
                        Remain:0,
                        Remains:0,
                        Remaining:0
                    },
                    Future:{
                        Return:-1,
                        "Will remain":0,
                        "Will be remaining":0,
                        "Will have remained":0
                    }
                },
                Seem:{
                    Return:-1,
                    Past:{
                        Return:-1,
                        Seemed:0,
                        "Was seeming":0,
                        "Had seemed":0
                    },
                    Present:{
                        Return:-1,
                        Seem:0,
                        Seems:0,
                        Seeming:0
                    },
                    Future:{
                        Return:-1,
                        "Will seem":0,
                        "Will be seeming":0,
                        "Will have seemed":0
                    }
                },
                "None of the above":{
                    Return:-1,
                    Smell:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Smelled:0,
                            "Was smelling":0,
                            "Had smelled":0
                        },
                        Present:{
                            Return:-1,
                            Smell:0,
                            Smells:0,
                            Smelling:0
                        },
                        Future:{
                            Return:-1,
                            "Will smell":0,
                            "Will be smelling":0,
                            "Will have smelled":0
                        }
                    },
                    Sound:{
                        Return:-1,
                        Past:{
                            Return:-1,
                            Sounded:0,
                            "Was sounding":0,
                            "Had sounded":0
                        },
                        Present:{
                            Return:-1,
                            Sound:0,
                            Sounds:0,
                            Sounding:0
                        },
                        Future:{
                            Return:-1,
                            "Will sound":0,
                            "Will be sounding":0,
                            "Will have sounded":0
                        }
                    },
                    "None of the above":{
                        Return:-1,
                        Stay:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Stayed:0,
                                "Was staying":0,
                                "Had stayed":0
                            },
                            Present:{
                                Return:-1,
                                Stay:0,
                                Stays:0,
                                Staying:0
                            },
                            Future:{
                                Return:-1,
                                "Will stay":0,
                                "Will be staying":0,
                                "Will have stayed":0
                            }
                        },
                        Taste:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Tasted:0,
                                "Was tasting":0,
                                "Had tasted":0
                            },
                            Present:{
                                Return:-1,
                                Taste:0,
                                Tastes:0,
                                Tasting:0
                            },
                            Future:{
                                Return:-1,
                                "Will taste":0,
                                "Will be tasting":0,
                                "Will have tasted":0
                            }
                        },
                        Turn:{
                            Return:-1,
                            Past:{
                                Return:-1,
                                Turned:0,
                                "Was turning":0,
                                "Had turned":0
                            },
                            Present:{
                                Return:-1,
                                Turn:0,
                                Turns:0,
                                Turning:0
                            },
                            Future:{
                                Return:-1,
                                "Will turn":0,
                                "Will be turning":0,
                                "Will have turned":0
                            }
                        }
                    }
                }
            }
        }
    },
};